import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Page Setup ---
st.set_page_config(page_title="PDF RAG App", page_icon="📄", layout="centered")
st.title("📄 Chat with PDF (Groq RAG)")

# --- 2. API Key Configuration ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = groq_api_key
except KeyError:
    st.error("⚠️ Please set your GROQ_API_KEY in the Streamlit secrets to proceed.")
    st.stop()

# --- 3. Sidebar: File Upload ---
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF here", type=["pdf"])

# --- 4. Main Application Logic ---
if uploaded_file is not None:
    if "vector_store" not in st.session_state:
        with st.spinner("Processing PDF and creating vector embeddings..."):
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            vector_store = FAISS.from_texts(chunks, embeddings)
            st.session_state.vector_store = vector_store
            
        st.success("PDF processed successfully!")

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        temperature=0.1
    )

    retriever = st.session_state.vector_store.as_retriever()
    
    system_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # --- 5. Chat Interface ---
    st.divider()
    st.subheader("Ask questions about your document:")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("What is this document about?"):
        st.chat_message("user").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_question})
                answer = response["answer"]
                st.markdown(answer)
                
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Please upload a PDF document in the sidebar to get started.")
