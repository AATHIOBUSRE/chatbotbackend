from typing import List
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
import httpx
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
import sqlite3
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import re
from PyPDF2 import PdfReader
from fastapi.middleware.cors import CORSMiddleware
import time
from dotenv import load_dotenv
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint  # Use the new class
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
# Setup FastAPI
app = FastAPI()
# Enable CORS
origins = [
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# JWT Configuration
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
 
# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
 
# Database setup
DB_FILE = "chatbot.db"
if not os.path.exists(DB_FILE):
    connection = sqlite3.connect(DB_FILE)
    connection.execute("PRAGMA journal_mode=WAL;")
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    connection.commit()
    connection.close()
 
# Utility Functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)
 
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
 
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
 
def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
 
def get_user_from_db(username: str):
    with sqlite3.connect(DB_FILE) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id, username, hashed_password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user:
            return {"id": user[0], "username": user[1], "hashed_password": user[2]}
    return None
 
def save_chat_history(user_id: int, question: str, answer: str):
    retry_count = 5
    while retry_count > 0:
        try:
            with sqlite3.connect(DB_FILE) as connection:
                cursor = connection.cursor()
                cursor.execute(
                    "INSERT INTO chat_history (user_id, question, answer) VALUES (?, ?, ?)",
                    (user_id, question, answer),
                )
                connection.commit()
            return
        except sqlite3.OperationalError:
            retry_count -= 1
            time.sleep(0.1)
    raise HTTPException(status_code=500, detail="Database is locked")
 
def clean_text(text: str):
    """Basic cleaning of text to improve embedding quality."""
    text = re.sub(r"\s+", " ", text)  # Remove excessive whitespace
    text = text.replace("\n", " ").strip()
    text = re.sub(r"(?<=\w) (?=\w@\w)", "", text)  # Fix spaces before email '@'
    return text
 
async def get_current_user(authorization: str = Header(...)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = get_user_from_db(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
 
# Request Models
class RegisterRequest(BaseModel):
    name: str
    email: str
    username: str
    password: str
 
class LoginRequest(BaseModel):
    username: str
    password: str
 
class AskQuestionRequest(BaseModel):
    question: str
    pdf_names: List[str] = None
 
# Endpoints
@app.post("/register")
async def register(request: RegisterRequest):
    connection = None
    try:
        hashed_password = hash_password(request.password)
        with sqlite3.connect(DB_FILE) as connection:
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, username, hashed_password) VALUES (?, ?, ?, ?)",
                (request.name, request.email, request.username, hashed_password),
            )
            connection.commit()
        return {"message": "User registered successfully."}
    except sqlite3.IntegrityError as e:
        if connection:
            connection.rollback()
        error_message = str(e)
        if "email" in error_message:
            raise HTTPException(status_code=400, detail="Email ID already exists")
        elif "username" in error_message:
            raise HTTPException(status_code=400, detail="Username already exists")
        else:
            raise HTTPException(status_code=400, detail="An error occurred during registration")
    finally:
        if connection:
            connection.close()
 
@app.post("/login")
async def login(request: LoginRequest):
    user = get_user_from_db(request.username)
    if not user or not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user["username"], "user_id": user["id"]})
    return {"access_token": access_token, "token_type": "bearer","userName":user["username"]}

Settings = {
    "llm": HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/google/gemma-1.1-7b-it",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),  # Ensure API key is loaded
        max_new_tokens=512,  # Pass explicitly
        temperature=0.1,  # Pass explicitly
    ),
    "embed_model": HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    ),
}


@app.post("/process-pdf/")
async def process_pdf(files: List[UploadFile] = File(...), current_user: dict = Depends(get_current_user)):
    try:
        processed_files = []
        embeddings = Settings["embed_model"]
    
        for pdf in files:
            text = ""
            pdf_reader = PdfReader(pdf.file)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text.strip() + "\n"

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
            chunks = text_splitter.split_text(text)

            if not chunks:
                continue

            vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
            pdf_name = os.path.splitext(pdf.filename)[0]
            faiss_file = f"faiss_index_{current_user['id']}_{pdf_name}"
            vector_store.save_local(faiss_file)

            processed_files.append(pdf_name)

        return {"message": "Uploaded Successfully", "processed_files": processed_files}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask-question/")
async def ask_question(request: dict, current_user: dict = Depends(get_current_user)):
    try:
        if "query" not in request:
            raise HTTPException(status_code=400, detail="Missing 'query' parameter in request body.")

        question_text = request["query"]
        
        # Find all FAISS indices belonging to the user
        indices = [f for f in os.listdir() if f.startswith(f"faiss_index_{current_user['id']}_")]
        
        if not indices:
            raise HTTPException(status_code=404, detail="No FAISS index found. Process a PDF first.")

        # Sort indices based on modification time (newest first)
        indices.sort(key=os.path.getmtime, reverse=True)

        # Select the latest FAISS index
        latest_index = indices[0]

        embeddings = Settings["embed_model"]
        vector_store = FAISS.load_local(latest_index, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()

        docs = retriever.invoke(question_text)
        combined_context = " ".join([doc.page_content for doc in docs])

        # === Enhanced Prompt: Generate Summarized Answers ===
        prompt_template = """
You are an AI assistant providing **clear and summarized answers** based on the given context.

- Extract relevant details accurately.
- Provide a **short but informative summary** instead of a one-word answer.
- If a specific detail is missing, simply respond with **"Not available"** instead of mentioning that the text does not include it.

Context: {context}
Question: {question}
Summarized Answer:
"""
        model = Settings["llm"]
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt})
        raw_answer = qa_chain.invoke({"query": question_text})

        refined_answer = raw_answer.get("result", "Not available").strip()

        return {"question": question_text, "answer": refined_answer}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/clear-history/")
async def clear_history(current_user: dict = Depends(get_current_user)):
    """
    Clear the chat history for the logged-in user.
    """
    try:
        user_id = current_user["id"]

        # Connect to the database and clear history
        with sqlite3.connect(DB_FILE) as connection:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
            connection.commit()

        return {"message": "Chat history cleared successfully."}
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail="Database error: " + str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred: " + str(e))
