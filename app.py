import streamlit as st
import os
from dotenv import load_dotenv

# =========================
# 🔐 ENV
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Falta GROQ_API_KEY en .env")
    st.stop()

# =========================
# 📚 LangChain
# =========================
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# =========================
# 🌐 URL Loader (ROBUSTO)
# =========================
import requests
from bs4 import BeautifulSoup

def load_url_content(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # eliminar scripts/estilos
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n")

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if not clean_text:
            return []

        return [Document(page_content=clean_text)]

    except Exception as e:
        st.error(f"Error cargando URL: {e}")
        return []

# =========================
# 🎨 UI
# =========================
st.set_page_config(page_title="RAG PRO FINAL", layout="wide")
st.title("🧠 RAG PRO FINAL (PDF + URL + MEMORIA)")

# =========================
# 🔄 SESSION STATE
# =========================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "source_hash" not in st.session_state:
    st.session_state.source_hash = None

# =========================
# 🔄 RESET
# =========================
def reset_all():
    st.session_state.vectorstore = None
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.source_hash = None

if st.button("🔄 Reset total"):
    reset_all()
    st.success("Reiniciado completamente ✅")

# =========================
# 🎯 SELECTOR DE FUENTE
# =========================
source_type = st.radio("Selecciona fuente", ["PDF/TXT", "URL"])

documents = None
current_hash = None

# =========================
# 📄 FILE
# =========================
if source_type == "PDF/TXT":
    uploaded_file = st.file_uploader("Sube archivo", type=["pdf", "txt"])

    if uploaded_file:
        current_hash = uploaded_file.name + str(uploaded_file.size)

        if current_hash != st.session_state.source_hash:
            reset_all()
            st.session_state.source_hash = current_hash

        file_path = f"temp_{uploaded_file.name}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents = loader.load()

# =========================
# 🌐 URL
# =========================
elif source_type == "URL":
    url = st.text_input("Pega una URL")

    if url:
        current_hash = url

        if current_hash != st.session_state.source_hash:
            reset_all()
            st.session_state.source_hash = current_hash

        documents = load_url_content(url)

        st.write(f"Contenido extraído: {len(documents)} documento(s)")

        if not documents:
            st.error("No se pudo extraer contenido de la URL ❌")
            st.stop()

# =========================
# 🧠 PROCESAMIENTO
# =========================
if documents and not st.session_state.vectorstore:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = splitter.split_documents(documents)
    texts = [t for t in texts if t.page_content.strip() != ""]

    if not texts:
        st.error("Documento sin contenido útil ❌")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(texts, embeddings)

    st.session_state.vectorstore = vectorstore
    st.success("Vector DB creada 🚀")

# =========================
# 💬 CHAT
# =========================
query = st.chat_input("Haz una pregunta...")

if query and st.session_state.vectorstore:

    # Guardar user
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # =========================
    # 🧠 MEMORIA
    # =========================
    history = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages[-6:]
    ])

    # =========================
    # 🤖 LLM
    # =========================
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
Eres un asistente experto.

Historial:
{history}

Contexto:
{context}

Pregunta:
{question}

Responde SOLO usando el contexto.
Si no está en el contexto, dilo claramente.
""")

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "history": history,
        "context": context,
        "question": query
    })

    # Guardar respuesta
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # UI
    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("assistant", answer))

# =========================
# 🖥️ CHAT UI
# =========================
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)