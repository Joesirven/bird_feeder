# global dependencies for the backend with SQLModel and JWT tokens authentication

from sqlmodel import SQLModel, Session, select
from sqlalchemy import create_engine
from pydantic import BaseModel
from typing import List, Optional
