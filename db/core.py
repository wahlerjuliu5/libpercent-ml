from fastapi import APIRouter
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase



load_dotenv()

router = APIRouter(prefix="/train")

url = os.getenv("TURSO_DATABASE_URL")
auth_token = os.getenv("TURSO_AUTH_TOKEN")
dbUrl = f"sqlite+{url}/?authToken={auth_token}&secure=true"
engine = create_engine(dbUrl, connect_args={'check_same_thread': False}, echo=True)



class NotFoundError(Exception):
    pass


class Base(DeclarativeBase):
    pass



session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Dependency to get the database session
def get_db():
    database = session_local()
    try:
        yield database
    finally:
        database.close()