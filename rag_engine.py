import re
import os
import pickle
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INDEX_FILE = "data/tfidf_index.pkl"


def read_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_docx(file):
    document = Document(file)
    text = ""
    for para in document.paragraphs:
        if para.text:
            text += para.text + "\n"
    return text


def read_txt(file):
    if hasattr(file, "read"):
        raw = file.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return raw
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    if file_name.endswith(".docx"):
        return read_docx(uploaded_file)
    if file_name.endswith(".txt"):
        return read_txt(uploaded_file)
    return ""


def clean_text(text):
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def chunk_text(text, chunk_size=180, overlap=40):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += max(1, chunk_size - overlap)

    return chunks


def create_tfidf_index(chunks):
    if not chunks:
        return None, None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 2),
        max_features=20000
    )
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def save_index(chunks, vectorizer, matrix):
    os.makedirs("data", exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "vectorizer": vectorizer,
            "matrix": matrix
        }, f)


def load_index():
    if not os.path.exists(INDEX_FILE):
        return [], None, None

    try:
        with open(INDEX_FILE, "rb") as f:
            data = pickle.load(f)
        return data["chunks"], data["vectorizer"], data["matrix"]
    except Exception:
        return [], None, None


def retrieve_context(query, chunks, vectorizer, matrix, top_k=4):
    if not query or not chunks or vectorizer is None or matrix is None:
        return []

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()
    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        if scores[idx] > 0:
            results.append(chunks[idx])

    return results


def build_index_from_upload_folder(upload_dir="data/uploads"):
    all_text = ""
    if not os.path.exists(upload_dir):
        return [], None, None

    for filename in os.listdir(upload_dir):
        path = os.path.join(upload_dir, filename)
        if filename.lower().endswith(".txt"):
            all_text += read_txt(path) + "\n"

    cleaned = clean_text(all_text)
    chunks = chunk_text(cleaned)
    vectorizer, matrix = create_tfidf_index(chunks)

    if chunks and vectorizer is not None:
        save_index(chunks, vectorizer, matrix)

    return chunks, vectorizer, matrix
