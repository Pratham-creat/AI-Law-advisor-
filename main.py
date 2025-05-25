import uvicorn
import logging
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.hash import bcrypt
import httpx
import pytesseract

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Create FastAPI app instance
app = FastAPI()

# Set up static directory for serving static files (like HTML, images)
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Database setup using SQLite and SQLAlchemy ORM
DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# User model for authentication
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

# Create database tables
Base.metadata.create_all(bind=engine)

# Serve the login page
@app.get("/", response_class=HTMLResponse)
async def serve_login():
    try:
        with open(os.path.join(static_dir, "login.html")) as f:
            return f.read()
    except FileNotFoundError:
        logging.error("Frontend file 'login.html' not found.")
        raise HTTPException(status_code=404, detail="Login page not found.")

# Register a new user
@app.post("/register")
async def register_user(username: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    hashed_password = bcrypt.hash(password)
    user = User(username=username, password=hashed_password)
    try:
        db.add(user)
        db.commit()
        return HTMLResponse(content="<h2>Registration successful. Please <a href='/'>login</a>.</h2>", status_code=201)
    except Exception as e:
        db.rollback()
        logging.error(f"Registration error: {e}")
        return HTMLResponse(content="<h2>Username already exists. Please try again.</h2>", status_code=400)
    finally:
        db.close()

# Handle user login
@app.post("/login")
async def handle_login(username: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    db.close()
    if user and bcrypt.verify(password, user.password):
        return RedirectResponse(url="/law", status_code=302)
    return HTMLResponse(content="<h2>Invalid credentials. Please try again.</h2>", status_code=401)

# Serve the main law advisor page
@app.get("/law", response_class=HTMLResponse)
async def serve_law():
    try:
        with open(os.path.join(static_dir, "law.html")) as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Law page not found.")

# Enable CORS for all origins and methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Fallback legal FAQ dictionary for common legal topics
LEGAL_FAQ = {
    "eviction": "If you are being evicted, ensure the landlord has given a proper notice (typically 30 days). You have the right to contest it in court if it's unjust.",
    "termination": "You may be entitled to severance or notice depending on your contract. Termination without cause may be challengeable under labor laws.",
    "rent": "Rent increases must follow your state laws and usually require 30 days’ notice.",
    "security deposit": "Landlords must return your security deposit within a certain time after you move out, often 30 days.",
    "notice period": "The standard notice period for resignation or eviction depends on state law or your contract—typically 30 days."
}

# Fetch relevant legal context based on keywords in the user's question
def fetch_relevant_legal_context(question: str) -> tuple:
    for keyword, context in LEGAL_FAQ.items():
        if keyword in question.lower():
            return keyword, context
    return None, "Refer to the Indian Constitution and local civil laws for more details."

# Call the local Ollama LLM server with a prompt and return the response
def call_llm(prompt: str) -> str | None:
    try:
        logging.info(f"Prompt to Ollama: {prompt}")
        ollama_url = "http://localhost:11434/api/generate"  # Make sure the port matches your Ollama config
        payload = {
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(ollama_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
    except Exception as e:
        logging.error(f"Ollama error: {e}")
        return None

# Endpoint to handle legal questions from the user
@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "").strip()

        if not question:
            raise HTTPException(status_code=400, detail="No question provided.")

        keyword, context = fetch_relevant_legal_context(question)
        prompt = f"{context}\n\nUser question: {question}\n\nAnswer:"
        answer = call_llm(prompt)

        # Fallback to FAQ if LLM fails and keyword is found
        if not answer and keyword:
            return {"answer": f"(response) {LEGAL_FAQ[keyword]}"}
        elif not answer:
            return {"answer": "(Fallback) Unable to process the question. Please consult a legal advisor."}

        return {"answer": answer}
    except Exception as e:
        logging.error(f"/ask error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Endpoint to handle document (image) upload and summarize its content
@app.post("/upload-doc")
async def summarize_document(file: UploadFile = File(...)):
    try:
        # Only allow PNG or JPEG images
        if file.content_type not in ["image/png", "image/jpeg"]:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PNG or JPEG.")

        # Read image content and extract text using OCR
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image)

        # Summarize the extracted text using the LLM
        prompt = f"Summarize this legal document in simple terms:\n\n{text}"
        summary = call_llm(prompt)

        if not summary:
            return {"summary": "(Fallback) Unable to summarize at the moment."}

        return {"summary": summary}
    except Exception as e:
        logging.error(f"OCR error: {e}")
        raise HTTPException(status_code=500, detail="Error processing document.")

# Run the FastAPI app with Uvicorn when this script is executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

    
    
    
    #to run this program --> pythom -m uvicorn main:app --reload
