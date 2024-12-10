from fastapi import FastAPI, HTTPException, Form, Header, Depends
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
import sqlite3
import os
from typing import List, Optional

# Setup FastAPI
app = FastAPI()

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

def get_user_from_db(username: str) -> Optional[dict]:
    with sqlite3.connect(DB_FILE) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id, username, hashed_password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user:
            return {"id": user[0], "username": user[1], "hashed_password": user[2]}
    return None

def get_user_by_id(user_id: int) -> Optional[dict]:
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()
    cursor.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    connection.close()
    if user:
        return {"id": user[0], "username": user[1]}
    return None

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
        connection = sqlite3.connect(DB_FILE)
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
            raise HTTPException(status_code=400, detail="Email already exists")
        elif "username" in error_message:
            raise HTTPException(status_code=400, detail="Username already exists")
        else:
            raise HTTPException(status_code=400, detail="An error occurred during registration")
    finally:
        if connection:
            connection.close()

@app.post("/token")
async def login(username: str = Form(...), password: str = Form(...)):
    user = get_user_from_db(username)
    if not user or not verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user["username"], "user_id": user["id"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me")
async def get_user_details(current_user: dict = Depends(get_current_user)):
    return current_user

@app.get("/chat-history")
async def chat_history(current_user: dict = Depends(get_current_user)):
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()
    cursor.execute("SELECT question, answer, timestamp FROM chat_history WHERE user_id = ?", (current_user["id"],))
    history = cursor.fetchall()
    connection.close()
    return [{"question": q, "answer": a, "timestamp": t} for q, a, t in history]
