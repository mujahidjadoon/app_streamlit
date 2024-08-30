import streamlit as st
import os
import sys

st.set_page_config(page_title="Assignment 1: RAG Q&A System", page_icon="🤖")
st.title("Assignment 1: RAG Pipeline Q&A System")

# Function to check if a package is installed
def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

# List of required packages
required_packages = [
    "langchain_community",
    "langchain",
    "chromadb",
    "sentence_transformers",
    "pypdf"
]

# Check for missing packages
missing_packages = [pkg for pkg in required_packages if not is_package_installed(pkg)]

if missing_packages:
    st.error(f"The following required packages are missing: {', '.join(missing_packages)}")
    st.info("Please contact the administrator to install the missing packages.")
    st.stop()

# If all packages are available, proceed with imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Rest of your Streamlit app code here...
# (Include your existing code for setting up the RAG pipeline, handling user input, etc.)

if __name__ == "__main__":
    st.success("Script loaded successfully")
