from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import uvicorn
from typing import List

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# Helper functions
def get_pdf_text(pdf_docs: List[UploadFile]):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context and with your own knowledge, make sure to provide all the details. If the answer is not in the 
    provided context, try to give the correct answer with your knowledge but make sure you don't provide the wrong answer.

    Context: \n {context}\n
    Question: \n {question} \n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@app.post("/process-pdf/")
async def process_pdf(files: List[UploadFile] = File(...)):
    try:
        raw_text = get_pdf_text(files)
        text_chunks = get_text_chunks(raw_text)
        get_vectorstore(text_chunks)
        return JSONResponse(content={"message": "PDF processed successfully."}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/ask-question/")
async def ask_question(question: str = Form(...)):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(question)
        chain = get_conversation_chain()
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return JSONResponse(content={"question":question, "answer": response["output_text"]}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
