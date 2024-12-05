from fastapi import FastAPI, Depends, HTTPException, Form, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import List
import sqlite3
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
 
# Setup FastAPI and configurations
app = FastAPI()
 
# JWT Configuration
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
 
# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
 
# Configure Google GenAI
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
 
# Database setup
DB_FILE = "chatbot.db"
if not os.path.exists(DB_FILE):
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, hashed_password TEXT NOT NULL)")
    cursor.execute("CREATE TABLE chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, question TEXT, answer TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id))")
    connection.commit()
    connection.close()
 
# Utility functions
def hash_password(password: str):
    return pwd_context.hash(password)
 
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)
 
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
 
def get_user_from_db(username: str):
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()
    cursor.execute("SELECT id, username, hashed_password FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    connection.close()
    if user:
        return {"id": user[0], "username": user[1], "hashed_password": user[2]}
    return None
 
def save_chat_history(user_id: int, question: str, answer: str):
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO chat_history (user_id, question, answer) VALUES (?, ?, ?)", (user_id, question, answer))
    connection.commit()
    connection.close()
 
def get_chat_history(user_id: int):
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()
    cursor.execute("SELECT question, answer, timestamp FROM chat_history WHERE user_id = ?", (user_id,))
    history = cursor.fetchall()
    connection.close()
    return [{"question": q, "answer": a} for q, a, t in history]
 
def get_chat_history_by_date(user_id: int, date: str):
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()
    cursor.execute("SELECT question, answer, timestamp FROM chat_history WHERE user_id = ? AND DATE(timestamp) = ?", (user_id, date))
    history = cursor.fetchall()
    connection.close()
    return [{"question": q, "answer": a} for q, a, t in history]
 
# Endpoint for user registration
@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    try:
        hashed_password = hash_password(password)
        connection = sqlite3.connect(DB_FILE)
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (username, hashed_password))
        connection.commit()
        connection.close()
        return {"message": "User registered successfully."}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
 
# Endpoint for login and token generation
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_from_db(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}
 
# Helper to decode token and get current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = get_user_from_db(username)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
 
# Process PDF and store embeddings
@app.post("/process-pdf/")
async def process_pdf(files: List[UploadFile] = File(...), current_user: dict = Depends(get_current_user)):
    try:
        # Extract text from PDFs
        text = ""
        for pdf in files:
            from PyPDF2 import PdfReader
            pdf_reader = PdfReader(pdf.file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        # Chunk text and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=300)
        chunks = text_splitter.split_text(text)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        vector_store.save_local(f"faiss_index_{current_user['id']}")
        return {"message": "PDF processed successfully for user."}
    except Exception as e:
        return {"error": str(e)}
 
# Chat endpoint
@app.post("/ask-question/")
async def ask_question(question: str = Form(None), current_user: dict = Depends(get_current_user)):
    try:
        if not question:
            # Retrieve chat history if no question provided
            history = get_chat_history(current_user["id"])
            return {"message": "Previous chat history retrieved.", "history": history}
 
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(f"faiss_index_{current_user['id']}", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(question)
 
        # Setup conversation chain
        prompt_template = """
        Context: {context}
        Question: {question}
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
 
        # Save and return chat history
        save_chat_history(current_user["id"], question, response["output_text"])
        history = get_chat_history(current_user["id"])
        return {"question": question, "answer": response["output_text"], "history": history}
    except Exception as e:
        return {"error": str(e)}
 
# Endpoint to get chat history by date
# Endpoint to get all chat history grouped by date
@app.get("/chat-history-by-date/")
async def chat_history_grouped_by_date(current_user: dict = Depends(get_current_user)):
    try:
        connection = sqlite3.connect(DB_FILE)
        cursor = connection.cursor()
        cursor.execute(
            "SELECT question, answer, DATE(timestamp) as chat_date FROM chat_history WHERE user_id = ? ORDER BY timestamp",
            (current_user["id"],)
        )
        history = cursor.fetchall()
        connection.close()
        
        if not history:
            return {"message": "No chat history found."}

        # Group chat history by date
        grouped_history = {}
        for question, answer, chat_date in history:
            if chat_date not in grouped_history:
                grouped_history[chat_date] = []
            grouped_history[chat_date].append({"question": question, "answer": answer})
        
        return {"history_by_date": grouped_history}
    except Exception as e:
        return {"error": str(e)}
