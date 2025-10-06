# app.py

import os
import json
import torch
import streamlit as st
from pathlib import Path
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

JSON_DIRECTORY_PATH = Path("./data/parsed")
FAISS_INDEX_PATH = "faiss_index_nasa"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "google/gemma-1.1-2b-it"

st.set_page_config(
    page_title="NASA Bioscience RAG Explorer",
    page_icon="🚀",
    layout="wide"
)

with st.sidebar:
    st.header("🚀 About")
    st.markdown(
        "This application is a smart search engine for NASA's bioscience research. "
        "It uses a Retrieval-Augmented Generation (RAG) system to answer questions "
        "based on a collection of 608 scientific publications."
    )
    st.divider()
    st.subheader("Example Questions")
    st.info("How does microgravity affect plant growth?")
    st.info("What is the role of the ACE2 receptor in viral infections?")
    st.info("What are the effects of space radiation on DNA?")

@st.cache_resource
def load_vector_db():
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        with st.spinner("Building vector index for the first time... Please wait."):
            docs = []
            json_files = [f for f in os.listdir(JSON_DIRECTORY_PATH) if f.endswith(".json")]
            for filename in json_files:
                file_path = os.path.join(JSON_DIRECTORY_PATH, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        page_content = data.get("All_text")
                        if page_content:
                            metadata = {
                                "source": data.get("url", "N/A"),
                                "title": data.get("title", "No Title"),
                                "doi": data.get("doi", "N/A")
                            }
                            docs.append(Document(page_content=page_content, metadata=metadata))
                except Exception as e:
                    st.error(f"Error processing file {filename}: {e}")
            if not docs:
                st.error("No documents were loaded.")
                return None
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': False},
            )
            vector_db = FAISS.from_documents(chunks, embeddings)
            vector_db.save_local(FAISS_INDEX_PATH)
            return vector_db

@st.cache_resource
def load_llm_and_chain():
    llm_pipeline_obj = pipeline(
        "text-generation",
        model=LLM_MODEL,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline_obj)
    
    prompt_template = """
    ### Instruction:
    Answer the question based only on the context provided below.
    Provide a concise and direct answer.
    Do not include any citation numbers or references like [42] or [57] in your response.

    ### Context:
    {context}

    ### Question:
    {question}

    ### Answer:
    """
    custom_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    vector_db = load_vector_db()
    if vector_db:
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt}
        )
        return qa_chain
    return None

st.title("🛰️ NASA Bioscience Research Explorer")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you explore NASA's bioscience research today?"}]

chain = load_llm_and_chain()

if chain:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Searching for answers..."):
            try:
                result = chain.invoke({"query": prompt})
                
                full_response = result['result']
                clean_answer = full_response.split("### Answer:")[-1].strip()
                
                sources_text = "\n\n**Sources:**\n"
                if 'source_documents' in result:
                    unique_docs = []
                    seen_titles = set()
                    for doc in result['source_documents']:
                        title = doc.metadata.get('title', 'N/A')
                        if title not in seen_titles:
                            seen_titles.add(title)
                            unique_docs.append(doc)
                    
                    for doc in unique_docs:
                        title = doc.metadata.get('title', 'N/A')
                        url = doc.metadata.get('source', '#')
                        sources_text += f"- **{title}** ([Link]({url}))\n"

                response_content = clean_answer + sources_text

                with st.chat_message("assistant"):
                    st.markdown(response_content)

                st.session_state.messages.append({"role": "assistant", "content": response_content})

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.error("Failed to initialize the RAG system. Please check the logs.")