from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from db.dbmodels import PastEntries
import pandas as pd



async def get_training_data(session: Session):
    df = pd.read_sql_query(select(PastEntries).where(PastEntries.id >= 1000), session.bind)
    session.close()
    return df