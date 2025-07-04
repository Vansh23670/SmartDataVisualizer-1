"""
Database service functions for Crypto Trading Dashboard
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func
import logging

from database import (
    SessionLocal, CryptoPrice, Portfolio, TradingSignal, 
    SentimentData, APIStatus, init_database
)

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service class for database operations"""
    
    def __init__(self):
        """Initialize database service"""
        init_database()
    
    def get_session(self) -> Session:
        """Get database session"""
        return SessionLocal()
    
    # Price Data Operations
    def save_crypto_price(self, symbol: str, price: float, volume: float = None, source: str = "api") -> bool:
        """Save cryptocurrency price to database"""
        try:
            with self.get_session() as db:
                price_entry = CryptoPrice(
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    source=source,
                    timestamp=datetime.utcnow()
                )
                db.add(price_entry)
                db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save price for {symbol}: {e}")
            return False
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from database"""
        try:
            with self.get_session() as db:
                latest = db.query(CryptoPrice).filter(
                    CryptoPrice.symbol == symbol
                ).order_by(desc(CryptoPrice.timestamp)).first()
                return latest.price if latest else None
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            return None
    
    def get_price_history(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get price history for specified time period"""
        try:
            with self.get_session() as db:
                since = datetime.utcnow() - timedelta(hours=hours)
                prices = db.query(CryptoPrice).filter(
                    and_(
                        CryptoPrice.symbol == symbol,
                        CryptoPrice.timestamp >= since
                    )
                ).order_by(CryptoPrice.timestamp).all()
                
                return [
                    {
                        "timestamp": price.timestamp,
                        "price": price.price,
                        "volume": price.volume
                    }
                    for price in prices
                ]
        except Exception as e:
            logger.error(f"Failed to get price history for {symbol}: {e}")
            return []
    
    # Portfolio Operations
    def update_portfolio(self, crypto_symbol: str, amount: float, average_price: float = None) -> bool:
        """Update portfolio holdings"""
        try:
            with self.get_session() as db:
                portfolio = db.query(Portfolio).filter(
                    Portfolio.crypto_symbol == crypto_symbol
                ).first()
                
                if portfolio:
                    portfolio.amount = amount
                    if average_price:
                        portfolio.average_price = average_price
                    portfolio.last_updated = datetime.utcnow()
                else:
                    portfolio = Portfolio(
                        crypto_symbol=crypto_symbol,
                        amount=amount,
                        average_price=average_price,
                        last_updated=datetime.utcnow()
                    )
                    db.add(portfolio)
                
                db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update portfolio for {crypto_symbol}: {e}")
            return False
    
    def get_portfolio(self) -> Dict[str, Dict]:
        """Get current portfolio holdings"""
        try:
            with self.get_session() as db:
                holdings = db.query(Portfolio).all()
                return {
                    holding.crypto_symbol: {
                        "amount": holding.amount,
                        "average_price": holding.average_price,
                        "last_updated": holding.last_updated
                    }
                    for holding in holdings
                }
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
            return {}
    
    # Trading Signals
    def save_trading_signal(self, symbol: str, signal_type: str, indicator: str, 
                          value: float, confidence: float = 0.5) -> bool:
        """Save trading signal to database"""
        try:
            with self.get_session() as db:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    indicator=indicator,
                    value=value,
                    confidence=confidence,
                    timestamp=datetime.utcnow()
                )
                db.add(signal)
                db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save trading signal: {e}")
            return False
    
    def get_recent_signals(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get recent trading signals"""
        try:
            with self.get_session() as db:
                since = datetime.utcnow() - timedelta(hours=hours)
                signals = db.query(TradingSignal).filter(
                    and_(
                        TradingSignal.symbol == symbol,
                        TradingSignal.timestamp >= since
                    )
                ).order_by(desc(TradingSignal.timestamp)).all()
                
                return [
                    {
                        "signal_type": signal.signal_type,
                        "indicator": signal.indicator,
                        "value": signal.value,
                        "confidence": signal.confidence,
                        "timestamp": signal.timestamp
                    }
                    for signal in signals
                ]
        except Exception as e:
            logger.error(f"Failed to get signals for {symbol}: {e}")
            return []
    
    # Sentiment Data
    def save_sentiment_data(self, symbol: str, sentiment_score: float, 
                           positive: float, negative: float, neutral: float, 
                           source_count: int = 0) -> bool:
        """Save sentiment analysis data"""
        try:
            with self.get_session() as db:
                sentiment = SentimentData(
                    symbol=symbol,
                    sentiment_score=sentiment_score,
                    positive=positive,
                    negative=negative,
                    neutral=neutral,
                    source_count=source_count,
                    timestamp=datetime.utcnow()
                )
                db.add(sentiment)
                db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save sentiment data: {e}")
            return False
    
    def get_sentiment_history(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get sentiment history"""
        try:
            with self.get_session() as db:
                since = datetime.utcnow() - timedelta(days=days)
                sentiments = db.query(SentimentData).filter(
                    and_(
                        SentimentData.symbol == symbol,
                        SentimentData.timestamp >= since
                    )
                ).order_by(SentimentData.timestamp).all()
                
                return [
                    {
                        "sentiment_score": sent.sentiment_score,
                        "positive": sent.positive,
                        "negative": sent.negative,
                        "neutral": sent.neutral,
                        "source_count": sent.source_count,
                        "timestamp": sent.timestamp
                    }
                    for sent in sentiments
                ]
        except Exception as e:
            logger.error(f"Failed to get sentiment history: {e}")
            return []
    
    # API Status Tracking
    def log_api_status(self, api_name: str, endpoint: str, status_code: int, 
                      response_time: float, is_working: bool, error_message: str = None) -> bool:
        """Log API status and performance"""
        try:
            with self.get_session() as db:
                api_status = APIStatus(
                    api_name=api_name,
                    endpoint=endpoint,
                    status_code=status_code,
                    response_time=response_time,
                    is_working=is_working,
                    error_message=error_message,
                    last_checked=datetime.utcnow()
                )
                db.add(api_status)
                db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to log API status: {e}")
            return False
    
    def get_api_reliability(self, hours: int = 24) -> Dict[str, Dict]:
        """Get API reliability statistics"""
        try:
            with self.get_session() as db:
                since = datetime.utcnow() - timedelta(hours=hours)
                
                # Get API statistics
                stats = db.query(
                    APIStatus.api_name,
                    func.count(APIStatus.id).label('total_requests'),
                    func.sum(func.cast(APIStatus.is_working, func.Integer())).label('successful_requests'),
                    func.avg(APIStatus.response_time).label('avg_response_time')
                ).filter(
                    APIStatus.last_checked >= since
                ).group_by(APIStatus.api_name).all()
                
                return {
                    stat.api_name: {
                        "total_requests": stat.total_requests,
                        "successful_requests": stat.successful_requests or 0,
                        "success_rate": (stat.successful_requests or 0) / stat.total_requests if stat.total_requests > 0 else 0,
                        "avg_response_time": float(stat.avg_response_time or 0)
                    }
                    for stat in stats
                }
        except Exception as e:
            logger.error(f"Failed to get API reliability: {e}")
            return {}
    
    # Analytics and Insights
    def get_price_analytics(self, symbol: str, days: int = 30) -> Dict:
        """Get price analytics for a cryptocurrency"""
        try:
            with self.get_session() as db:
                since = datetime.utcnow() - timedelta(days=days)
                
                prices = db.query(CryptoPrice).filter(
                    and_(
                        CryptoPrice.symbol == symbol,
                        CryptoPrice.timestamp >= since
                    )
                ).order_by(CryptoPrice.timestamp).all()
                
                if not prices:
                    return {}
                
                price_values = [p.price for p in prices]
                
                return {
                    "symbol": symbol,
                    "period_days": days,
                    "min_price": min(price_values),
                    "max_price": max(price_values),
                    "avg_price": sum(price_values) / len(price_values),
                    "price_change": price_values[-1] - price_values[0] if len(price_values) > 1 else 0,
                    "price_change_percent": ((price_values[-1] - price_values[0]) / price_values[0] * 100) if len(price_values) > 1 and price_values[0] != 0 else 0,
                    "data_points": len(price_values),
                    "first_recorded": prices[0].timestamp,
                    "last_recorded": prices[-1].timestamp
                }
        except Exception as e:
            logger.error(f"Failed to get price analytics: {e}")
            return {}

# Global database service instance
db_service = DatabaseService()