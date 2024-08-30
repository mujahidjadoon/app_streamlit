import streamlit as st
import os
import sys

# Check for required packages
required_packages = [
    "langchain_community",
    "langchain",
    "chromadb",
    "sentence_transformers",
    "pypdf"
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    st.error(f"The following required packages are missing: {', '.join(missing_packages)}")
    st.info("Please install the missing packages before running this app.")
    st.stop()

# Now import the required modules
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Assignment 1: RAG Q&A System", page_icon="🤖")
st.title("Assignment 1: RAG Pipeline Q&A System")
st.markdown("[Access the deployed app here](https://rag-qa-system.streamlit.app/)")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your HuggingFace API Key:", type="password")
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    st.warning("Please enter your HuggingFace API Key in the sidebar to proceed.")
else:
    @st.cache_resource
    def load_documents():
        loader = PyPDFDirectoryLoader("./pdfs")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_documents(documents)

    with st.spinner("Loading documents..."):
        chunks = load_documents()

    @st.cache_resource
    def setup_rag():
        embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        vector_retriever = vectorstore.as_retriever()
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            model_kwargs={
                "max_new_tokens": 512,
                "top_k": 30,
                "temperature": 0.1,
                "repetition_penalty": 1.1,
                "return_full_text": False
            },
        )
        prompt_template = """
        <|system|>
        You are an AI Assistant that follows instructions extremely well.
        Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT
        CONTEXT: {context}
        </s>
        <|user|>
        {query}
        </s>
        <|assistant|>
        Your answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        output_parser = StrOutputParser()
        rag_chain = (
            {"context": ensemble_retriever, "query": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
        )
        return rag_chain

    with st.spinner("Setting up RAG pipeline..."):
        rag_chain = setup_rag()

    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke(query)
        st.write("Answer:", response)

# Add information about required files
st.sidebar.markdown("---")
st.sidebar.subheader("Required Files")
st.sidebar.info(
    "Make sure you have a 'pdfs' folder in the same directory as this script, "
    "containing the PDF documents you want to use for the RAG system."
)

# Add information about deployment
st.sidebar.markdown("---")
st.sidebar.subheader("Deployment")
st.sidebar.info(
    "To deploy this app, make sure you have a 'requirements.txt' file "
    "in your repository with all the necessary dependencies listed."
)

if __name__ == "__main__":
    print("Script executed successfully")
