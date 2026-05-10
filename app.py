import os
import time
import requests
import pandas as pd
import streamlit as st
from langdetect import detect

# Disable Streamlit watcher issues
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

from database import (
    init_db,
    get_all_users,
    update_user_role,
    save_chat,
    get_chat_history,
    create_escalation_ticket,
    get_escalation_tickets,
    update_ticket,
    save_knowledge_file,
    get_knowledge_files,
    save_evaluation_result,
    get_evaluation_results
)

from auth import register_user, login_user

from rag_engine import (
    extract_text,
    clean_text,
    chunk_text,
    create_tfidf_index,
    retrieve_context,
    save_index,
    load_index,
    build_index_from_upload_folder
)

from evaluation import calculate_similarity_score, classify_accuracy


# ============================================================
# Initial Setup
# ============================================================

init_db()

UPLOAD_DIR = "data/uploads"
EVALUATION_DIR = "data/evaluation"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

st.set_page_config(
    page_title="Enterprise AI Chatbot Platform",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# Session State
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    chunks, vectorizer, matrix = load_index()
    st.session_state.chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.matrix = matrix

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "matrix" not in st.session_state:
    st.session_state.matrix = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_ollama_error" not in st.session_state:
    st.session_state.last_ollama_error = ""


# ============================================================
# Helper Functions
# ============================================================

def detect_language(text):
    try:
        language_code = detect(text)
        if language_code == "ar":
            return "Arabic"
        return "English"
    except Exception:
        if any("\u0600" <= ch <= "\u06FF" for ch in text):
            return "Arabic"
        return "English"


def safety_filter(user_input):
    blocked_terms = [
        "private key",
        "credit card",
        "bypass security",
        "malware",
        "exploit",
        "steal data"
    ]

    lowered = user_input.lower()

    for term in blocked_terms:
        if term in lowered:
            return False

    return True


def check_ollama_server():
    """
    Checks if Ollama local server is reachable.
    """
    try:
        response = requests.get(
            "http://127.0.0.1:11434/api/tags",
            timeout=10
        )

        if response.status_code == 200:
            return True, response.json()

        return False, f"Ollama server returned status code {response.status_code}: {response.text}"

    except Exception as e:
        return False, str(e)


def get_ollama_models():
    """
    Gets available Ollama models from local API.
    """
    ok, result = check_ollama_server()

    if not ok:
        return [], result

    try:
        models = result.get("models", [])
        model_names = [m.get("name", "") for m in models if m.get("name")]
        return model_names, ""
    except Exception as e:
        return [], str(e)


def normalize_model_name(model_name):
    """
    Makes model name more compatible with Ollama.
    """
    model_name = str(model_name).strip()

    if model_name == "llama3":
        return "llama3"

    if model_name == "llama3:latest":
        return "llama3"

    if model_name == "mistral:latest":
        return "mistral"

    if model_name == "llama2:latest":
        return "llama2"

    return model_name


def build_prompt(user_question, retrieved_context, language, chatbot_role):
    context_text = "\n\n".join(retrieved_context).strip()

    if not context_text:
        context_text = "No uploaded knowledge-base context is available."

    if language == "Arabic":
        instruction = f"""
أنت مساعد ذكاء اصطناعي مؤسسي.

دورك: {chatbot_role}

التزم بالتعليمات التالية:
- أجب باللغة العربية الواضحة والمهنية.
- إذا وُجد سياق من قاعدة المعرفة، استخدمه في الإجابة.
- إذا لم توجد معلومات كافية في الملفات، قل ذلك بوضوح.
- لا تخترع سياسات أو أرقام داخلية غير موجودة في السياق.
- أعطِ إجابة عملية ومنظمة.
"""
    else:
        instruction = f"""
You are an enterprise AI assistant.

Your role is: {chatbot_role}

Rules:
- Answer in clear professional English.
- Use the knowledge-base context when available.
- If the files do not contain enough information, say so clearly.
- Do not invent internal policies, numbers, or claims.
- Give a practical and structured answer.
"""

    prompt = f"""
{instruction}

Knowledge Base Context:
{context_text}

User Question:
{user_question}

Answer:
"""

    return prompt


def fallback_response(user_question, retrieved_context, language, ollama_error=None):
    """
    Safe fallback so the chatbot never stays silent.
    """

    if language == "Arabic":
        if retrieved_context:
            answer = "أنا أعمل الآن في وضع احتياطي لأن Ollama لا يرد من داخل التطبيق.\n\n"
            answer += "لكنني وجدت معلومات قريبة من قاعدة المعرفة:\n\n"

            for i, chunk in enumerate(retrieved_context[:3], start=1):
                answer += f"المصدر {i}:\n{chunk[:700]}...\n\n"

            answer += "الخلاصة: راجعي المصادر أعلاه، وإذا كان السؤال يحتاج قراراً رسمياً يمكن تصعيده لمراجع بشري."
        else:
            answer = """
أنا أعمل الآن في وضع احتياطي لأن Ollama لا يرد من داخل التطبيق.

لا توجد معلومات كافية من قاعدة المعرفة للإجابة بدقة.

يمكنك:
- رفع ملف PDF أو DOCX أو TXT
- إعادة بناء الفهرس
- استخدام نموذج أخف مثل gemma2:2b
- تصعيد السؤال لمراجع بشري
"""
    else:
        if retrieved_context:
            answer = "I am currently in fallback mode because Ollama is not responding inside the app.\n\n"
            answer += "However, I found related content from the knowledge base:\n\n"

            for i, chunk in enumerate(retrieved_context[:3], start=1):
                answer += f"Source {i}:\n{chunk[:700]}...\n\n"

            answer += "Summary: Please review the retrieved sources above. If this requires an official decision, escalate it to a human reviewer."
        else:
            answer = """
I am currently in fallback mode because Ollama is not responding inside the app.

There is not enough uploaded knowledge-base content to answer accurately.

You can:
- Upload a PDF, DOCX, or TXT file
- Rebuild the knowledge index
- Use a lighter model such as gemma2:2b
- Escalate the question to a human reviewer
"""

    if ollama_error:
        answer += f"\n\nTechnical Ollama error:\n{ollama_error}"

    return answer.strip()


def generate_response(prompt, model_name, temperature, user_question, retrieved_context, language):
    """
    Main response function.
    Uses direct Ollama HTTP API instead of ollama Python package.
    """

    model_name = normalize_model_name(model_name)

    try:
        url = "http://127.0.0.1:11434/api/chat"

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": 150
            }
        }

        response = requests.post(
            url,
            json=payload,
            timeout=180
        )

        if response.status_code != 200:
            error_message = (
                f"Ollama API status code: {response.status_code}\n\n"
                f"Ollama response:\n{response.text}"
            )

            st.session_state.last_ollama_error = error_message

            return fallback_response(
                user_question,
                retrieved_context,
                language,
                error_message
            )

        data = response.json()

        if "message" in data:
            message = data["message"]

            if isinstance(message, dict) and "content" in message:
                answer = message["content"]

                if answer and answer.strip():
                    st.session_state.last_ollama_error = ""
                    return answer.strip()

        error_message = f"Unexpected Ollama response format:\n{data}"
        st.session_state.last_ollama_error = error_message

        return fallback_response(
            user_question,
            retrieved_context,
            language,
            error_message
        )

    except requests.exceptions.Timeout:
        error_message = (
            "Ollama connected but took too long to answer. "
            "Try using gemma2:2b instead of llama3."
        )

        st.session_state.last_ollama_error = error_message

        return fallback_response(
            user_question,
            retrieved_context,
            language,
            error_message
        )

    except requests.exceptions.ConnectionError as e:
        error_message = (
            "Cannot connect to Ollama API at http://127.0.0.1:11434\n\n"
            "Ollama may not be running as a local API server.\n\n"
            f"Details: {e}"
        )

        st.session_state.last_ollama_error = error_message

        return fallback_response(
            user_question,
            retrieved_context,
            language,
            error_message
        )

    except Exception as e:
        error_message = f"Unexpected error while calling Ollama:\n{e}"

        st.session_state.last_ollama_error = error_message

        return fallback_response(
            user_question,
            retrieved_context,
            language,
            error_message
        )


# ============================================================
# Login Page
# ============================================================

def login_page():
    st.title("🔐 Enterprise AI Chatbot Login")

    st.info(
        "Register a user first. After registration, you can make yourself admin "
        "by running: python make_admin.py"
    )

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        st.subheader("Login")

        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            success, user, message = login_user(username, password)

            if success:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success(message)
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(message)

    with tab_register:
        st.subheader("Register New User")

        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")

        preferred_language = st.selectbox(
            "Preferred Language",
            ["English", "Arabic"]
        )

        if st.button("Create Account"):
            success, message = register_user(
                new_username,
                new_password,
                role="user",
                preferred_language=preferred_language
            )

            if success:
                st.success(message)
            else:
                st.error(message)


# ============================================================
# Main App
# ============================================================

def main_app():
    user = st.session_state.user

    username = user["username"]
    role = user["role"]

    st.sidebar.title("🤖 Enterprise AI Chatbot")
    st.sidebar.write(f"User: **{username}**")
    st.sidebar.write(f"Role: **{role}**")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.messages = []
        st.rerun()

    st.sidebar.markdown("---")

    chatbot_role = st.sidebar.selectbox(
        "Chatbot Role",
        [
            "HR Assistant",
            "Training Assistant",
            "Customer Support Assistant",
            "Medical Engineering Assistant",
            "General Enterprise Assistant"
        ]
    )

    available_models, model_error = get_ollama_models()

    if available_models:
        default_index = 0

        for i, model in enumerate(available_models):
            if model.startswith("gemma2:2b"):
                default_index = i
                break

        model_name = st.sidebar.selectbox(
            "Ollama Model",
            available_models,
            index=default_index
        )
    else:
        st.sidebar.warning("Could not read Ollama models. Type model manually.")
        model_name = st.sidebar.text_input(
            "Ollama Model",
            value="gemma2:2b"
        )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3
    )

    top_k = st.sidebar.slider(
        "Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=4
    )

    enable_safety = st.sidebar.checkbox(
        "Enable Safety Filter",
        value=True
    )

    st.sidebar.markdown("---")

    if st.sidebar.button("Test Ollama Connection"):
        ok, result = check_ollama_server()

        if ok:
            st.sidebar.success("Ollama server is reachable.")

            test_prompt = "Say hello in one short sentence."

            test_answer = generate_response(
                test_prompt,
                model_name,
                temperature,
                test_prompt,
                [],
                "English"
            )

            st.sidebar.write("Test answer:")
            st.sidebar.write(test_answer)
        else:
            st.sidebar.error("Ollama server is not reachable.")
            st.sidebar.write(result)

    if st.session_state.last_ollama_error:
        with st.sidebar.expander("Last Ollama Error"):
            st.write(st.session_state.last_ollama_error)

    menu = [
        "Chatbot",
        "Knowledge Base",
        "Escalation",
        "Evaluation"
    ]

    if role in ["admin", "manager"]:
        menu.append("Admin Dashboard")

    page = st.sidebar.radio("Navigation", menu)

    if page == "Chatbot":
        chatbot_page(
            username,
            chatbot_role,
            model_name,
            temperature,
            top_k,
            enable_safety
        )

    elif page == "Knowledge Base":
        knowledge_base_page(username)

    elif page == "Escalation":
        escalation_page(username, role)

    elif page == "Evaluation":
        evaluation_page(
            model_name,
            temperature,
            chatbot_role,
            top_k
        )

    elif page == "Admin Dashboard":
        admin_dashboard_page()


# ============================================================
# Chatbot Page
# ============================================================

def chatbot_page(username, chatbot_role, model_name, temperature, top_k, enable_safety):
    st.title("💬 Enterprise AI Chatbot")

    if not st.session_state.chunks:
        st.warning(
            "No knowledge base loaded yet. "
            "Upload documents in the Knowledge Base page for document-based answers."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask your question in English or Arabic...")

    if user_question:
        language = detect_language(user_question)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.write(user_question)

        if enable_safety and not safety_filter(user_question):
            bot_answer = (
                "I cannot help with requests involving private credentials, "
                "harmful actions, or unsafe activity."
            )
            retrieved_context = []

        else:
            retrieved_context = retrieve_context(
                user_question,
                st.session_state.chunks,
                st.session_state.vectorizer,
                st.session_state.matrix,
                top_k=top_k
            )

            st.session_state.last_sources = retrieved_context

            prompt = build_prompt(
                user_question,
                retrieved_context,
                language,
                chatbot_role
            )

            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    bot_answer = generate_response(
                        prompt,
                        model_name,
                        temperature,
                        user_question,
                        retrieved_context,
                        language
                    )

                st.write(bot_answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": bot_answer
                }
            )

            save_chat(
                username,
                user_question,
                bot_answer,
                language
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("👍 Helpful"):
                    st.success("Feedback recorded for this session.")

            with col2:
                if st.button("🚨 Escalate to Human"):
                    create_escalation_ticket(
                        username,
                        user_question,
                        bot_answer
                    )
                    st.warning("Question escalated to a human reviewer.")

            with st.expander("Retrieved Knowledge Sources"):
                if retrieved_context:
                    for i, chunk in enumerate(retrieved_context, start=1):
                        st.markdown(f"**Source Chunk {i}:**")
                        st.write(chunk[:1200])
                        st.markdown("---")
                else:
                    st.info("No source chunks retrieved.")


# ============================================================
# Knowledge Base Page
# ============================================================

def knowledge_base_page(username):
    st.title("📂 Knowledge Base Management")

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload TXT, PDF, or DOCX files",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process and Index Uploaded Files"):
        all_text = ""

        for file in uploaded_files:
            extracted_text = extract_text(file)
            cleaned_text = clean_text(extracted_text)

            txt_filename = file.name + ".txt"
            txt_path = os.path.join(UPLOAD_DIR, txt_filename)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            save_knowledge_file(
                file.name,
                username,
                "Manual Upload"
            )

            all_text += cleaned_text + "\n"

        chunks = chunk_text(all_text)
        vectorizer, matrix = create_tfidf_index(chunks)

        st.session_state.chunks = chunks
        st.session_state.vectorizer = vectorizer
        st.session_state.matrix = matrix

        if chunks and vectorizer is not None:
            save_index(chunks, vectorizer, matrix)
            st.success(
                f"Knowledge base indexed successfully with {len(chunks)} chunks."
            )
        else:
            st.error("No text could be extracted from the uploaded files.")

    if st.button("Rebuild Index from Saved Uploads"):
        chunks, vectorizer, matrix = build_index_from_upload_folder(UPLOAD_DIR)

        st.session_state.chunks = chunks
        st.session_state.vectorizer = vectorizer
        st.session_state.matrix = matrix

        if chunks:
            st.success(
                f"Index rebuilt successfully with {len(chunks)} chunks."
            )
        else:
            st.warning("No saved TXT knowledge files found.")

    st.markdown("---")

    st.subheader("Uploaded Knowledge Files")

    files = get_knowledge_files()

    if files:
        df = pd.DataFrame(
            files,
            columns=[
                "ID",
                "Filename",
                "Uploaded By",
                "Source Type",
                "Created At"
            ]
        )

        st.dataframe(df, use_container_width=True)
    else:
        st.info("No files uploaded yet.")

    st.markdown("---")

    st.subheader("SharePoint / Google Drive Placeholder")

    st.info(
        "Paste a SharePoint or Google Drive link to register the source. "
        "Full API sync can be added later."
    )

    source_option = st.selectbox(
        "External Source",
        ["SharePoint", "Google Drive"]
    )

    external_link = st.text_input(
        "Folder or Document Link"
    )

    if st.button("Register External Source"):
        if external_link:
            save_knowledge_file(
                external_link,
                username,
                source_option
            )

            st.success(f"{source_option} source registered.")
        else:
            st.warning("Please paste a link.")


# ============================================================
# Escalation Page
# ============================================================

def escalation_page(username, role):
    st.title("🚨 Human Escalation Workflow")

    tickets = get_escalation_tickets()

    if not tickets:
        st.info("No escalation tickets yet.")
        return

    df = pd.DataFrame(
        tickets,
        columns=[
            "Ticket ID",
            "Username",
            "Question",
            "Bot Response",
            "Status",
            "Assigned To",
            "Admin Note",
            "Created At",
            "Updated At"
        ]
    )

    if role == "user":
        df = df[df["Username"] == username]

    st.dataframe(df, use_container_width=True)

    if role in ["admin", "manager"]:
        st.markdown("---")

        st.subheader("Update Ticket")

        ticket_id = st.number_input(
            "Ticket ID",
            min_value=1,
            step=1
        )

        status = st.selectbox(
            "Status",
            [
                "Open",
                "In Review",
                "Resolved",
                "Closed"
            ]
        )

        assigned_to = st.text_input("Assigned To")

        admin_note = st.text_area(
            "Admin Note / Human Answer"
        )

        if st.button("Update Ticket"):
            update_ticket(
                ticket_id,
                status,
                assigned_to,
                admin_note
            )

            st.success("Ticket updated.")


# ============================================================
# Evaluation Page
# ============================================================

def evaluation_page(model_name, temperature, chatbot_role, top_k):
    st.title("📊 Chatbot Accuracy Testing")

    st.write(
        "Upload a CSV with two columns: `question` and `expected_answer`."
    )

    uploaded_eval = st.file_uploader(
        "Upload Evaluation CSV",
        type=["csv"]
    )

    if uploaded_eval:
        eval_df = pd.read_csv(uploaded_eval)

        if not {"question", "expected_answer"}.issubset(eval_df.columns):
            st.error("CSV must contain: question, expected_answer")
            return

        st.dataframe(eval_df, use_container_width=True)

        if st.button("Run Accuracy Test"):
            results = []

            for _, row in eval_df.iterrows():
                question = str(row["question"])
                expected_answer = str(row["expected_answer"])

                language = detect_language(question)

                retrieved_context = retrieve_context(
                    question,
                    st.session_state.chunks,
                    st.session_state.vectorizer,
                    st.session_state.matrix,
                    top_k=top_k
                )

                prompt = build_prompt(
                    question,
                    retrieved_context,
                    language,
                    chatbot_role
                )

                bot_answer = generate_response(
                    prompt,
                    model_name,
                    temperature,
                    question,
                    retrieved_context,
                    language
                )

                score = calculate_similarity_score(
                    expected_answer,
                    bot_answer
                )

                rating = classify_accuracy(score)

                save_evaluation_result(
                    question,
                    expected_answer,
                    bot_answer,
                    score
                )

                results.append(
                    {
                        "Question": question,
                        "Expected Answer": expected_answer,
                        "Bot Answer": bot_answer,
                        "Similarity Score": score,
                        "Rating": rating
                    }
                )

            result_df = pd.DataFrame(results)

            st.subheader("Evaluation Results")
            st.dataframe(result_df, use_container_width=True)

            if not result_df.empty:
                st.metric(
                    "Average Similarity Score",
                    round(result_df["Similarity Score"].mean(), 3)
                )

    st.markdown("---")

    st.subheader("Previous Evaluation Results")

    previous = get_evaluation_results()

    if previous:
        previous_df = pd.DataFrame(
            previous,
            columns=[
                "Question",
                "Expected Answer",
                "Bot Answer",
                "Score",
                "Created At"
            ]
        )

        st.dataframe(previous_df, use_container_width=True)
    else:
        st.info("No previous evaluation results.")


# ============================================================
# Admin Dashboard
# ============================================================

def admin_dashboard_page():
    st.title("🛠️ Admin Dashboard")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Users",
            "Roles",
            "Chat History",
            "Knowledge Files",
            "System KPIs"
        ]
    )

    with tab1:
        st.subheader("Registered Users")

        users = get_all_users()

        if users:
            df = pd.DataFrame(
                users,
                columns=[
                    "ID",
                    "Username",
                    "Role",
                    "Preferred Language",
                    "Created At"
                ]
            )

            st.dataframe(df, use_container_width=True)
        else:
            st.info("No users found.")

    with tab2:
        st.subheader("Change User Role")

        username = st.text_input("Username to update")

        role = st.selectbox(
            "New Role",
            [
                "user",
                "manager",
                "admin"
            ]
        )

        if st.button("Update Role"):
            update_user_role(username, role)
            st.success(
                "User role updated. Ask the user to logout and login again."
            )

    with tab3:
        st.subheader("Chat History")

        history = get_chat_history()

        if history:
            df = pd.DataFrame(
                history,
                columns=[
                    "Username",
                    "User Message",
                    "Bot Response",
                    "Language",
                    "Created At"
                ]
            )

            st.dataframe(df, use_container_width=True)
        else:
            st.info("No chat history found.")

    with tab4:
        st.subheader("Knowledge Files")

        files = get_knowledge_files()

        if files:
            df = pd.DataFrame(
                files,
                columns=[
                    "ID",
                    "Filename",
                    "Uploaded By",
                    "Source Type",
                    "Created At"
                ]
            )

            st.dataframe(df, use_container_width=True)
        else:
            st.info("No files found.")

    with tab5:
        st.subheader("System KPIs")

        users = get_all_users()
        history = get_chat_history()
        tickets = get_escalation_tickets()
        evaluations = get_evaluation_results()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Users", len(users))
        col2.metric("Total Conversations", len(history))
        col3.metric("Escalation Tickets", len(tickets))
        col4.metric("Evaluation Tests", len(evaluations))

        if evaluations:
            scores = [row[3] for row in evaluations]
            average_score = sum(scores) / len(scores)

            st.metric(
                "Average Accuracy Score",
                round(average_score, 3)
            )


# ============================================================
# Run App
# ============================================================

if not st.session_state.logged_in:
    login_page()
else:
    main_app()