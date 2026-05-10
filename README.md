# Enterprise AI Chatbot Platform - Fixed Version

This is a fixed Streamlit enterprise chatbot project.

It avoids the previous PyTorch / sentence-transformers problem by using TF-IDF retrieval instead of Torch-based embeddings.

## Features

- Login system
- User roles: user, manager, admin
- SQLite database storage
- Chat history
- Knowledge-base upload: PDF, DOCX, TXT
- TF-IDF retrieval system
- Optional Ollama response generation
- Built-in fallback response when Ollama does not answer
- Admin dashboard
- Role management
- Human escalation workflow
- Evaluation dataset upload
- Accuracy similarity testing
- Arabic-English support
- SharePoint / Google Drive placeholder

## Install

Open terminal inside the project folder:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## First Login

1. Open the app.
2. Register a new user.
3. Login with that user.

## Make Yourself Admin

After registering your first user, stop Streamlit or open a new terminal in the same folder and run:

```bash
python make_admin.py
```

Enter your username.

Then logout and login again.

## Optional Ollama Setup

This app can work even if Ollama does not respond.

To use Ollama:

```bash
ollama pull llama3.2
ollama serve
```

Then use this model name in the sidebar:

```text
llama3.2
```

If Ollama still fails, the app will automatically answer in fallback mode.

## Knowledge Base

Go to:

```text
Knowledge Base > Upload Documents > Process and Index Uploaded Files
```

Upload PDF, DOCX, or TXT files.

Then go to Chatbot and ask questions.

## Evaluation Dataset

Use the included file:

```text
sample_evaluation_dataset.csv
```

It must contain:

```text
question, expected_answer
```

## Why This Version Should Work Better

The previous version used `sentence-transformers`, which loads PyTorch. On some Windows + Streamlit setups, this causes:

```text
RuntimeError: Tried to instantiate class '__path__._path'
```

This version uses:

```text
scikit-learn TF-IDF
```

So it does not require PyTorch, FAISS, or sentence-transformers.

## Notes

- TF-IDF is simpler than vector embeddings but reliable for training and demos.
- For production, you can later replace TF-IDF with ChromaDB, FAISS, or Azure AI Search.
- The fallback answer is intentional, so the chatbot never appears silent.
