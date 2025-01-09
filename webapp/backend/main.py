# fastapi with routers, cors, and uvicorn, global dependencies, and pydantic models & queries using SQLModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Session, select
from sqlalchemy import create_engine
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
