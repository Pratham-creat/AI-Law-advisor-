import uvicorn
from together import Together
import logging
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import pytesseract
from PIL import Image
import io
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.hash import bcrypt
import requests
import json

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Initialize Together AI client with the API key
client = Together(api_key="YOUR_ACTUAL_API_KEY")  # Replace with your valid API key

# Create FastAPI app
app = FastAPI()

# Static directory
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Database setup
DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

@app.get("/", response_class=HTMLResponse)
async def serve_login():
    try:
        with open(os.path.join(static_dir, "login.html")) as f:
            return f.read()
    except FileNotFoundError:
        logging.error("Frontend file 'login.html' not found in the static directory.")
        raise HTTPException(status_code=404, detail="Login page not found.")

@app.post("/register")
async def register_user(username: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    hashed_password = bcrypt.hash(password)  # Password is hashed before storing
    user = User(username=username, password=hashed_password)
    try:
        db.add(user)
        db.commit()
        db.refresh(user)
        return HTMLResponse(content="<h2>Registration successful. Please <a href='/'>login</a>.</h2>", status_code=201)
    except Exception as e:
        db.rollback()
        logging.error(f"Error during registration: {e}")
        return HTMLResponse(content="<h2>Username already exists. Please try again.</h2>", status_code=400)
    finally:
        db.close()

@app.post("/login")
async def handle_login(username: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    db.close()
    if user and bcrypt.verify(password, user.password):  # Password is verified
        logging.info("Login successful")
        return RedirectResponse(url="/law", status_code=302)
    logging.info("Login failed")
    return HTMLResponse(content="<h2>Invalid credentials. Please try again.</h2>", status_code=401)

@app.get("/law", response_class=HTMLResponse)
async def serve_law():
    try:
        with open(os.path.join(static_dir, "law.html")) as f:
            return f.read()
    except FileNotFoundError:
        logging.error("Frontend file 'law.html' not found in the static directory.")
        raise HTTPException(status_code=404, detail="Law page not found.")

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "").strip()

        if not question:
            raise HTTPException(status_code=400, detail="No question provided.")

        # Fetch relevant legal context
        keyword, context = fetch_relevant_legal_context(question)
        prompt = f"{context}\n\nUser question: {question}\n\nAnswer:"

        # Call the Ollama API
        answer = call_ollama(prompt)

        if not answer and keyword:
            logging.info(f"Using fallback answer for keyword: {keyword}")
            return {"answer": LEGAL_FAQ[keyword]}
        elif not answer:
            logging.info("Using generic fallback response.")
            return {"answer": "Unable to process the question. Please consult a legal advisor or refer to local laws."}

        return {"answer": answer}
    except Exception as e:
        logging.error(f"Error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Fallback legal FAQ
LEGAL_FAQ = {
    "eviction": "If you are being evicted, ensure the landlord has given a proper notice (typically 30 days). You have the right to contest it in court if it's unjust.",
    "termination": "You may be entitled to severance or notice depending on your contract. Termination without cause may be challengeable under labor laws.",
    "rent": "Rent increases must follow your state laws and usually require 30 days’ notice.",
    "security deposit": "Landlords must return your security deposit within a certain time after you move out, often 30 days.",
    "notice period": "The standard notice period for resignation or eviction depends on state law or your contract—typically 30 days."
}

def fetch_relevant_legal_context(question: str) -> tuple:
    for keyword, context in LEGAL_FAQ.items():
        if keyword in question.lower():
            return keyword, context
    return None, "Refer to the Indian Constitution and local civil laws for more details."

def call_llm(prompt: str) -> str | None:
    try:
        logging.info(f"Prompt sent to Together AI: {prompt}")
        response = client.generate(
            endpoint="google/gemma-3-27b-it",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        logging.info(f"Response from Together AI: {response}")
        return response.get("output", None)
    except Exception as e:
        logging.error(f"Together AI error: {e}")
        return None

def call_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "tinyllama",
                "messages": [
                    {"role": "system", "content": "You are a legal advisor."},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        response.raise_for_status()
        # Handle streaming JSON (one object per line)
        lines = response.text.strip().splitlines()
        last_json = json.loads(lines[-1])
        return last_json['message']['content']
    except Exception as e:
        logging.error(f"Ollama API error: {e}")
        logging.error(f"Ollama API raw response: {response.text}")
        return None