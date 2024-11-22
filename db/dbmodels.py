from sqlalchemy import String, Column, Integer, DateTime
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass

class PastEntries(Base):
    __tablename__ = "pastEntries"
    id = Column(Integer, primary_key=True)
    uniId = Column(String, nullable=True)
    createdAt = Column(String, nullable=False)
    iclPercentage = Column(Integer, nullable=False)
    currentTime = Column(Integer, nullable=False)
    currentDay = Column(Integer, nullable=False)
    currentDayofYear = Column(Integer, nullable=False)
    currentExamDelta = Column(Integer, nullable=False)
    temp = Column(Integer, nullable=False)
    wind = Column(Integer, nullable=False)
    rain = Column(Integer, nullable=False)
    rwkPercentage = Column(Integer, nullable=False)