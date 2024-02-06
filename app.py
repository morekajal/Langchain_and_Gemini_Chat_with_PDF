import streamlit as st 
from PyPDF2 import PdfReader
import os, io
from dotenv import load_dotenv

import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Upload PDf, Read Pdf, Convert to Vector Embeddings
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)   #will be list
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

import fitz  # PyMuPDF

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_file = io.BytesIO(pdf.read())  # Create a file-like object from bytes
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# divide the text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 500)
    chunks = text_splitter.split_text(text)
    return chunks 

# Convert text chunks into vectors
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     #this vector store can be saved in database or in local environment
#     vector_store.save_local("faiss_index")   # a folder of faiss_index will be created and inside will have embeddings, pickel file

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    encoded_text_chunks = [chunk.encode('utf-8', 'ignore').decode('utf-8') for chunk in text_chunks]
    vector_store = FAISS.from_texts(encoded_text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the questions as detailed as possible from provided context, make sure to provide all the details all the details,
    if the answer is not in the provided context just say, "Answer is not available in the Given Context".
    Don't provide wrong answers.
    Context : \n {context}? \n
    Question : \n {question} \n

    Answer :

    """
    # Initialize model
    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

    # create chain
    chain = load_qa_chain(model, chain_type="stuff", prompt = prompt)
    
    return chain

# w.r.t User Input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    print(response)

    st.write("Reply : ", response["output_text"])

# Streamlit app
def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat With Multiple PDF using Gemini")

    user_question = st.text_input("Ask Question from PDF Files Provided")

    if user_question:
        user_input(user_question)  # as soon as user inputs question, execute user_input function

    with st.sidebar:
        st.title("Menu :")
        pdf_docs = st.file_uploader("Upload your PDF files and Click on Submit and Process", accept_multiple_files=True)
        
        if pdf_docs is not None and st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done...")

if __name__ == "__main__":
    main()
         