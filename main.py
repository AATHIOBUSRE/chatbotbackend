from fastapi import FastAPI, Depends, HTTPException, Form, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import List
import sqlite3
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import re
from PyPDF2 import PdfReader

# Setup FastAPI
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
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        )
    """
    )
    cursor.execute("""
        CREATE TABLE chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """
    )
    connection.commit()
    connection.close()

# Utility Functions
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
    cursor.execute(
        "INSERT INTO chat_history (user_id, question, answer) VALUES (?, ?, ?)",
        (user_id, question, answer),
    )
    connection.commit()
    connection.close()

def clean_text(text: str):
    """Basic cleaning of text to improve embedding quality."""
    text = re.sub(r"\s+", " ", text)  # Remove excessive whitespace
    text = text.replace("\n", " ").strip()
    return text

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


# Endpoints
@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    try:
        hashed_password = hash_password(password)
        connection = sqlite3.connect(DB_FILE)
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
            (username, hashed_password),
        )
        connection.commit()
        connection.close()
        return {"message": "User registered successfully."}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_from_db(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

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

@app.post("/process-pdf/")
async def process_pdf(files: List[UploadFile] = File(...), current_user: dict = Depends(get_current_user)):
    try:
        processed_files = []
        
        for pdf in files:
            # Extract text from the PDF
            text = ""
            pdf_reader = PdfReader(pdf.file)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                text += clean_text(extracted_text)
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
            chunks = text_splitter.split_text(text)

            # Create embeddings for the chunks
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

            # Save the FAISS index with a unique name per PDF
            pdf_name = os.path.splitext(pdf.filename)[0]
            faiss_file = f"faiss_index_{current_user['id']}_{pdf_name}"
            vector_store.save_local(faiss_file)
            
            processed_files.append(pdf_name)

        return {"message": f"Processed and saved embeddings for PDFs: {processed_files}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask-question/")
async def ask_question(question: str = Form(...), pdf_names: List[str] = Form(None), current_user: dict = Depends(get_current_user)):
    try:
        # List to store relevant documents
        all_docs = []

        # If specific PDFs are mentioned, load only those indices
        if pdf_names:
            for pdf_name in pdf_names:
                faiss_file = f"faiss_index_{current_user['id']}_{pdf_name}"
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.load_local(faiss_file, embeddings, allow_dangerous_deserialization=True)
                docs = vector_store.similarity_search(question, k=5)
                all_docs.extend(docs)
        else:
            # Load all FAISS indices for the user if no PDF is specified
            indices = [f for f in os.listdir() if f.startswith(f"faiss_index_{current_user['id']}_")]
            for index in indices:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.load_local(index, embeddings, allow_dangerous_deserialization=True)
                docs = vector_store.similarity_search(question, k=5)
                all_docs.extend(docs)

        # Ensure documents are retrieved
        if not all_docs:
            return {"message": "No relevant documents found for the question."}

        # Setup conversation chain
        prompt_template = """
        Context: {context}
        Question: {question}
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        qa_chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

        # Combine contexts from documents and generate an answer
        combined_context = " ".join([doc.page_content for doc in all_docs])
        answer = qa_chain.run(input_documents=all_docs, question=question)

        # Save the question and answer to the user's chat history
        save_chat_history(current_user["id"], question, answer)

        return {"question": question, "answer": answer}
    except Exception as e:
        return {"error": str(e)}
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

