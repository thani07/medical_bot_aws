# db.py
import os
import uuid
from datetime import datetime

from sqlalchemy import create_engine, Column, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base

# --------------------------------------------------------------------
# REQUIRED: PostgreSQL connection via environment variable
#
# Example:
# DATABASE_URL = postgresql+psycopg2://admin:password@host:5432/chatbot
# --------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("ERROR: DATABASE_URL environment variable is not set.")

# SQLAlchemy engine for PostgreSQL
engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,       # prevents stale connection errors
    pool_recycle=1800         # handles RDS connection resets
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()

# --------------------------------------------------------------------
# MODELS
# --------------------------------------------------------------------
def gen_uuid():
    return str(uuid.uuid4())

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    title = Column(String(200), nullable=False, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# --------------------------------------------------------------------
# INIT FUNCTION (Creates tables in PostgreSQL)
# --------------------------------------------------------------------
def init_db():
    Base.metadata.create_all(bind=engine)


# --------------------------------------------------------------------
# CRUD OPERATIONS
# --------------------------------------------------------------------
def create_session(db, title="New Chat"):
    sess = ChatSession(title=title)
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess


def get_sessions(db):
    return db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()


def get_session(db, session_id):
    return db.query(ChatSession).filter(ChatSession.id == session_id).first()


def update_session_title(db, session_id, title):
    sess = get_session(db, session_id)
    if not sess:
        return None
    sess.title = title
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess


def add_message(db, session_id, role, content):
    msg = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_messages(db, session_id):
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )
