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
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/process-pdf/")
async def process_pdf(files: List[UploadFile] = File(...), current_user: dict = Depends(get_current_user)):
    try:
        processed_files = []

        for pdf in files:
            text = ""
            pdf_reader = PdfReader(pdf.file)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                text += clean_text(extracted_text)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
            chunks = text_splitter.split_text(text)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

            pdf_name = os.path.splitext(pdf.filename)[0]
            faiss_file = f"faiss_index_{current_user['id']}_{pdf_name}"
            vector_store.save_local(faiss_file)

            processed_files.append(pdf_name)

        return {"message": f"Processed and saved embeddings for PDFs: {processed_files}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask-question/")
async def ask_question(request: AskQuestionRequest, current_user: dict = Depends(get_current_user)):
    try:
        all_docs = []
        relevant_vector_store = None

        # Step 1: Retrieve relevant documents from FAISS vector stores
        indices = [f for f in os.listdir() if f.startswith(f"faiss_index_{current_user['id']}_")]
        for index in indices:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.load_local(index, embeddings, allow_dangerous_deserialization=True)
            docs = vector_store.similarity_search(request.question, k=5)
            all_docs.extend(docs)

            index_name = index.split(f"faiss_index_{current_user['id']}_")[1]
            if index_name.lower() in request.question.lower():
                relevant_vector_store = (vector_store, index)

        if not all_docs:
            return {"message": "No relevant documents found for the question."}

        # Step 2: Generate raw answer using QA Chain
        combined_context = " ".join([doc.page_content for doc in all_docs])
        prompt_template = """
        You are a helpful assistant. Answer the question based on the given context in a clear and professional manner.

        Context: {context}
        Question: {question}
        Improved Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        qa_chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
        raw_answer = qa_chain.run(input_documents=all_docs, question=request.question)

        print(f"Raw Answer from vector store: {raw_answer}")

        # Step 3: Refine raw answer using Gemini API
        api_key = "AIzaSyA7ac82_39rm88KGfPR0TtIE-TFni7RlNg"
        external_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

        request_body = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"""
                            You are a professional assistant. Given the question and its rough answer, provide a structured, grammatically correct response.

                            Question: {request.question}
                            Answer: {raw_answer.strip()}

                            Ensure the response is natural, professional, and includes necessary context or framing to sound complete and well-structured.
                            """
                        }
                    ]
                }
            ]
        }

        refined_answer = raw_answer.strip()  # Default to raw answer if refinement fails

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.post(
                    url=external_api_url,
                    headers={"Content-Type": "application/json"},
                    json=request_body
                )

                print(f"Request body sent to Gemini API: {request_body}")
                print(f"Response status: {response.status_code}")
                print(f"Response content: {response.text}")

                if response.status_code == 200:
                    # Parse response
                    refined_data = response.json()
                    # Check if the expected structure exists
                    parts = refined_data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        refined_answer = parts[0]["text"]
                        print(f"Refined answer: {refined_answer}")
                    else:
                        print("Text field not found in API response. Using raw answer.")
                else:
                    print(f"API returned error: {response.status_code} - {response.text}")
        except (httpx.RequestError, httpx.HTTPStatusError, Exception) as err:
            print(f"Error occurred during Gemini API call: {err}")

        # Step 4: Save chat history to the database
        save_chat_history(current_user["id"], request.question, refined_answer)

        # Step 5: Update vector store with the new Q&A pair
        if relevant_vector_store:
            vector_store, faiss_file = relevant_vector_store
            new_text = f"Q: {request.question} A: {refined_answer}"
            vector_store.add_texts([new_text])
            vector_store.save_local(faiss_file)

        # Step 6: Return the final structured answer
        return {"question": request.question, "answer": refined_answer}
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

        grouped_history = {}
        for question, answer, chat_date in history:
            if chat_date not in grouped_history:
                grouped_history[chat_date] = []
            grouped_history[chat_date].append({"question": question, "answer": answer})

        return {"history_by_date": grouped_history}
    except Exception as e:
        return {"error": str(e)}