import os
import subprocess
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# ------------------ Configuration ------------------

CHROMA_PATH = "chroma_db"
DOCUMENTS_PATH = "documents"
OLLAMA_MODEL = "mistral"

# ------------------ Chargement de documents et base vectorielle ------------------

@st.cache_resource
def build_vectorstore():
    loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    if not split_docs:
        return None

    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory=CHROMA_PATH)
    vectordb.persist()
    return vectordb

# ------------------ Requête à Ollama ------------------

def query_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )
        output = result.stdout.strip()
        return output if output else "⚠️ Réponse vide d’Ollama."

    except subprocess.TimeoutExpired:
        return "⏱️ Temps d'attente dépassé pour Ollama."
    except FileNotFoundError:
        return "❌ Ollama n’est pas installé ou accessible depuis ce script."
    except subprocess.CalledProcessError as e:
        if "not found" in e.stderr.lower():
            return f"❌ Le modèle '{model}' n'est pas installé. Utilisez `ollama pull {model}`"
        return f"⚠️ Erreur Ollama : {e.stderr.strip()}"
    except Exception as e:
        return f"🚨 Erreur inconnue avec Ollama : {str(e)}"

# ------------------ Initialisation de l'interface ------------------

st.set_page_config(page_title="Assistant Fonderie IA", layout="centered")
st.title("🤖 Assistant IA - Fonderie & Métallurgie")
question = st.text_input("Posez votre question :", placeholder="Ex. Quelle est la température de recuit pour l'acier X?")

# ------------------ Traitement principal ------------------

if question:
    vectordb = build_vectorstore()
    if vectordb:
        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=None,  # Pas d'OpenAI, on fait la logique manuellement
            retriever=retriever,
            return_source_documents=True
        )
        result = qa_chain.invoke({"query": question})
        sources = result.get("source_documents", [])
        answer = result.get("result", "").strip()

        if not sources or not answer or "je ne sais pas" in answer.lower():
            st.warning("Aucun document pertinent trouvé, utilisation d’Ollama Mistral.")
            answer = query_ollama(question)
        else:
            st.success("Réponse basée sur les documents chargés.")
    else:
        st.warning("Aucun document trouvé, utilisation d’Ollama Mistral.")
        answer = query_ollama(question)

    st.markdown("### 💬 Réponse")
    st.write(answer)
