from fastapi import FastAPI, Depends, HTTPException, Form, UploadFile, File, Header
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
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
from fastapi.middleware.cors import CORSMiddleware
import time

# Setup FastAPI
app = FastAPI()

# Enable CORS
origins = [
    "http://localhost",
    "http://localhost:4200",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

# Endpoints
@app.post("/register")
async def register(
    name: str = Form(...), 
    email: str = Form(...), 
    username: str = Form(...), 
    password: str = Form(...)
):
    connection = None
    try:
        hashed_password = hash_password(password)
        with sqlite3.connect(DB_FILE) as connection:
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, username, hashed_password) VALUES (?, ?, ?, ?)",
                (name, email, username, hashed_password),
            )
            connection.commit()
        return {"message": "User registered successfully."}
    except sqlite3.IntegrityError as e:
        if connection:
            connection.rollback()  # Rollback in case of errors
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
async def login(username: str = Form(...), password: str = Form(...)):
    user = get_user_from_db(username)
    if not user or not verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user["username"], "user_id": user["id"]})
    return {"access_token": access_token, "token_type": "bearer"}

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
async def ask_question(
    question: str = Form(...),
    pdf_names: List[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        all_docs = []
        relevant_vector_store = None  # Track the relevant vector store for updates

        # If specific PDFs are mentioned, load only those indices
        if pdf_names:
            for pdf_name in pdf_names:
                faiss_file = f"faiss_index_{current_user['id']}_{pdf_name}"
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.load_local(faiss_file, embeddings, allow_dangerous_deserialization=True)
                docs = vector_store.similarity_search(question, k=5)
                all_docs.extend(docs)

                # Save the vector store related to the mentioned PDF
                if pdf_name in question.lower():  # Match question topic with PDF name
                    relevant_vector_store = (vector_store, faiss_file)
        else:
            # Load all FAISS indices for the user if no PDF is specified
            indices = [f for f in os.listdir() if f.startswith(f"faiss_index_{current_user['id']}_")]
            for index in indices:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.load_local(index, embeddings, allow_dangerous_deserialization=True)
                docs = vector_store.similarity_search(question, k=5)
                all_docs.extend(docs)

                # Save the vector store related to the topic
                index_name = index.split(f"faiss_index_{current_user['id']}_")[1]
                if index_name.lower() in question.lower():
                    relevant_vector_store = (vector_store, index)

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

        # Add the generated answer to the relevant FAISS index
        if relevant_vector_store:
            vector_store, faiss_file = relevant_vector_store
            new_text = f"Q: {question} A: {answer}"
            vector_store.add_texts([new_text])  # Add new text to the vector store

            # Save the updated FAISS index only for the relevant folder
            vector_store.save_local(faiss_file)

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
