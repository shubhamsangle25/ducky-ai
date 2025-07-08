from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ducky_logic import get_ducky_response

app = FastAPI()

# CORS - allow FE JS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve static front end
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return FileResponse("frontend/index.html")


# message model
class Message(BaseModel):
    role: str
    content: str


class DuckyRequest(BaseModel):
    message: str
    mode: str
    history: List[Message]


@app.post("/talk")
async def talk_to_ducky(request: DuckyRequest):
    reply, updated_history = get_ducky_response(request.message, request.mode, request.history)
    return {"reply": reply, "history": updated_history}
