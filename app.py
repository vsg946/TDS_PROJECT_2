import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from solver import QuizSolver
from fastapi.responses import HTMLResponse, FileResponse

# ---------------------------------------------------------
# Main FastAPI app (ONLY ONE!)
# ---------------------------------------------------------
app = FastAPI()

# Root route (Render sends HEAD, so allow both GET + HEAD)
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def root():
    return "<html><head><title>Quiz Solver</title></head><body><h1>Quiz Solver is live 🎉</h1></body></html>"

# favicon (optional)
@app.get("/favicon.ico")
async def favicon():
    if os.path.exists("static/favicon.ico"):
        return FileResponse("static/favicon.ico")
    return HTMLResponse("")

# ---------------------------------------------------------
# Environment variables
# ---------------------------------------------------------
load_dotenv()
YOUR_SECRET = os.getenv("YOUR_SECRET", "")
YOUR_EMAIL = os.getenv("YOUR_EMAIL", "")

# ---------------------------------------------------------
# Quiz solving payload
# ---------------------------------------------------------
class QuizPayload(BaseModel):
    email: str
    secret: str
    url: str

# ---------------------------------------------------------
# QUIZ ENDPOINT
# ---------------------------------------------------------
@app.post("/quiz")
def receive_quiz(payload: QuizPayload):

    if payload.secret != YOUR_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    start = time.time()
    solver = QuizSolver(email=payload.email, secret=payload.secret)

    try:
        result = solver.solve_and_submit(payload.url, time_budget_sec=170)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    return {
        "status": "ok",
        "elapsed_sec": time.time() - start,
        "result": result
    }