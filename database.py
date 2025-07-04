"""
Database models and connection setup for Crypto Trading Dashboard
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import logging

logger = logging.getLogger(__name__)

# Database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL')

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=StaticPool,
    pool_pre_ping=True,
    echo=False
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

class CryptoPrice(Base):
    """Store historical cryptocurrency prices"""
    __tablename__ = "crypto_prices"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String(50))  # API source (coingecko, coinbase, etc.)
    
class Portfolio(Base):
    """Store portfolio holdings"""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    crypto_symbol = Column(String(20), nullable=False, index=True)
    amount = Column(Float, nullable=False, default=0.0)
    average_price = Column(Float)  # Average purchase price
    last_updated = Column(DateTime, default=datetime.utcnow)

class TradingSignal(Base):
    """Store trading signals and indicators"""
    __tablename__ = "trading_signals"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    signal_type = Column(String(20))  # buy, sell, hold
    indicator = Column(String(50))  # RSI, MACD, SMA, etc.
    value = Column(Float)
    confidence = Column(Float)  # Signal confidence (0-1)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class SentimentData(Base):
    """Store sentiment analysis data"""
    __tablename__ = "sentiment_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    sentiment_score = Column(Float)  # Compound sentiment score
    positive = Column(Float)
    negative = Column(Float)
    neutral = Column(Float)
    source_count = Column(Integer)  # Number of sources analyzed
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class APIStatus(Base):
    """Track API status and reliability"""
    __tablename__ = "api_status"
    
    id = Column(Integer, primary_key=True, index=True)
    api_name = Column(String(50), nullable=False, index=True)
    endpoint = Column(String(200))
    status_code = Column(Integer)
    response_time = Column(Float)  # Response time in seconds
    is_working = Column(Boolean, default=True)
    error_message = Column(Text)
    last_checked = Column(DateTime, default=datetime.utcnow, index=True)

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database with tables"""
    try:
        create_tables()
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    # Test database connection and create tables
    print(f"Connecting to database: {DATABASE_URL}")
    init_database()
    print("Database setup completed")