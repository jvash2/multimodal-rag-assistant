import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ==========================================
# CORE LOGIC: RAG ENGINE
# ==========================================
class RAGEngine:
    """Clase encargada de la gestión de documentos y base de datos vectorial."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def scrape_url(self, url):
        """Extrae contenido limpio de una URL."""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            
            text = soup.get_text(separator="\n")
            clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            
            return [Document(page_content=clean_text)] if clean_text else []
        except Exception as e:
            st.error(f"Error en Scraper: {e}")
            return []

    def create_vectorstore(self, documents):
        """Procesa documentos y genera la base vectorial Chroma."""
        texts = self.splitter.split_documents(documents)
        texts = [t for t in texts if t.page_content.strip() != ""]
        
        if not texts:
            return None
        
        return Chroma.from_documents(texts, self.embeddings)

# ==========================================
# UI & SESSION MANAGEMENT
# ==========================================
def initialize_session():
    """Inicializa los estados globales de la aplicación."""
    if "engine" not in st.session_state:
        st.session_state.engine = RAGEngine()
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "source_hash" not in st.session_state:
        st.session_state.source_hash = None

def reset_application():
    """Limpia todo el contexto actual."""
    st.session_state.vectorstore = None
    st.session_state.chat_history = []
    st.session_state.source_hash = None
    st.rerun()

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY in .env")
        st.stop()

    st.set_page_config(page_title="RAG Professional System", layout="wide")
    st.title("Advanced RAG Assistant")
    
    initialize_session()

    # Sidebar: Control y Carga
    with st.sidebar:
        st.header("Settings")
        if st.button("🔄 Reset System"):
            reset_application()
            
        source_type = st.radio("Knowledge Source", ["PDF/TXT", "URL"])
        documents = None
        current_id = None

        if source_type == "PDF/TXT":
            file = st.file_uploader("Upload Document", type=["pdf", "txt"])
            if file:
                current_id = f"{file.name}_{file.size}"
                # Lógica de persistencia temporal
                with open(f"temp_{file.name}", "wb") as f:
                    f.write(file.read())
                loader = PyPDFLoader(f"temp_{file.name}") if file.type == "application/pdf" else TextLoader(f"temp_{file.name}")
                documents = loader.load()

        elif source_type == "URL":
            url = st.text_input("Target URL")
            if url:
                current_id = url
                documents = st.session_state.engine.scrape_url(url)

        # Verificar si la fuente cambió para re-indexar
        if current_id and current_id != st.session_state.source_hash:
            st.session_state.source_hash = current_id
            with st.spinner("Indexing new source..."):
                st.session_state.vectorstore = st.session_state.engine.create_vectorstore(documents)
                st.success("Knowledge Base Updated")

    # Área de Chat
    if st.session_state.vectorstore:
        # Mostrar historial previo
        for role, text in st.session_state.chat_history:
            st.chat_message(role).write(text)

        query = st.chat_input("Ask a question about your data...")

        if query:
            st.chat_message("user").write(query)
            
            # Recuperación de contexto (Retrieval)
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
            context_docs = retriever.invoke(query)
            context_text = "\n\n".join([d.page_content for d in context_docs])
            
            # Construcción de Memoria
            history_str = "\n".join([f"{r}: {c}" for r, c in st.session_state.chat_history[-6:]])

            # Inferencia con Groq
            llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)
            
            prompt = ChatPromptTemplate.from_template("""
            Eres un asistente experto.
            Historial: {history}
            Contexto: {context}
            Pregunta: {question}
            Responde SOLO usando el contexto. Si no está, dilo claramente.
            """)

            chain = prompt | llm | StrOutputParser()
            
            with st.chat_message("assistant"):
                response = chain.invoke({
                    "history": history_str,
                    "context": context_text,
                    "question": query
                })
                st.write(response)
            
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("assistant", response))
    else:
        st.info("Please upload a file or enter a URL to start the AI session.")

if __name__ == "__main__":
    main()


# =========================
#  ADVERTISE !! the api key i used was free and i just selected temporaly for make the program works need create new api key!
# =========================