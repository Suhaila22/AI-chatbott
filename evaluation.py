from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity_score(expected_answer, bot_answer):
    expected_answer = str(expected_answer or "")
    bot_answer = str(bot_answer or "")

    if not expected_answer.strip() or not bot_answer.strip():
        return 0.0

    vectorizer = TfidfVectorizer().fit([expected_answer, bot_answer])
    vectors = vectorizer.transform([expected_answer, bot_answer])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(float(score), 3)


def classify_accuracy(score):
    if score >= 0.75:
        return "Excellent"
    if score >= 0.55:
        return "Good"
    if score >= 0.35:
        return "Needs Review"
    return "Poor"
