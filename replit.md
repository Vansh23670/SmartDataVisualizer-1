# Crypto Trading Dashboard

## Overview

This is a Streamlit-based cryptocurrency trading dashboard that provides real-time market data, technical analysis, sentiment analysis, and portfolio management capabilities. The application combines multiple data sources and analytical tools to create a comprehensive trading interface.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - provides the web interface and interactive components
- **Visualization**: Plotly for interactive charts and graphs
- **Layout**: Single-page application with multiple tabs/sections
- **Optional UI Enhancement**: streamlit-sortables for customizable dashboard widgets

### Backend Architecture
- **Runtime**: Python-based application running on Streamlit server
- **Data Processing**: Pandas for data manipulation and analysis
- **Async Operations**: aiohttp and asyncio for concurrent API calls
- **Threading**: Multi-threaded approach for real-time data updates

### Real-time Data Handling
- **WebSocket Connections**: For live cryptocurrency price feeds
- **Queue System**: Thread-safe data sharing between real-time feeds and UI
- **Async Processing**: Non-blocking API calls for multiple data sources

## Key Components

### 1. Data Sources & APIs
- **Cryptocurrency APIs**: Real-time price data via REST and WebSocket connections
- **Market Data**: Historical and live trading data
- **External Integrations**: Multiple API endpoints for comprehensive market coverage

### 2. Analytics Engine
- **Technical Analysis**: 
  - pandas_ta library for technical indicators (with fallback for compatibility issues)
  - Custom technical analysis implementations
- **Sentiment Analysis**: VADER sentiment analyzer for market sentiment
- **Machine Learning**: scikit-learn MLPRegressor for predictive modeling
- **Optimization**: scipy.optimize for portfolio optimization

### 3. Security & Data Management
- **Encryption**: Cryptography library (Fernet) for sensitive data protection
- **Portfolio Storage**: JSON-based portfolio persistence
- **Configuration Management**: Centralized CONFIG dictionary

### 4. Monitoring & Logging
- **Logging**: Python logging module for application monitoring
- **Error Handling**: Comprehensive exception handling for API failures and data issues

## Data Flow

1. **Real-time Data Ingestion**: WebSocket connections stream live price data
2. **Data Processing**: Raw data is processed through pandas and technical analysis pipelines
3. **Storage**: Processed data is stored in memory with optional persistence
4. **Visualization**: Plotly generates interactive charts from processed data
5. **User Interaction**: Streamlit handles user inputs and updates displays accordingly

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **requests/aiohttp**: HTTP client libraries
- **websocket-client**: WebSocket connectivity

### Analytics & ML
- **vaderSentiment**: Sentiment analysis
- **scikit-learn**: Machine learning algorithms
- **scipy**: Scientific computing and optimization
- **pandas_ta**: Technical analysis indicators (optional with fallback)

### Security & Utilities
- **cryptography**: Data encryption
- **nest_asyncio**: Async event loop handling
- **streamlit-sortables**: UI customization (optional)

## Deployment Strategy

### Local Development
- Standard Python environment with pip requirements
- Streamlit development server for testing
- Configuration through environment variables and JSON files

### Production Considerations
- **Scalability**: Single-user application architecture
- **Data Persistence**: File-based storage for portfolio and configuration
- **Security**: Encrypted sensitive data storage
- **Monitoring**: Application logging for debugging and monitoring

## Recent Changes

- July 04, 2025: Initial crypto dashboard setup with Streamlit
- July 04, 2025: Added PostgreSQL database integration with comprehensive models
  - Created database models for crypto prices, portfolio, trading signals, sentiment data, and API status
  - Integrated database service with existing portfolio management
  - Added database analytics section showing price history and API reliability
  - Updated data storage from encrypted JSON files to PostgreSQL database
  - Implemented automatic price and sentiment data logging to database
- July 04, 2025: Enhanced historical data fetching and UI improvements
  - Fixed CoinGecko API integration for reliable historical price data
  - Improved data sampling and timeframe handling for charts
  - Enhanced error messaging for optional features (blockchain, sentiment analysis)
  - Cleaned up status messages and removed unnecessary warnings
  - Maintained authentic data integrity throughout all features

## Changelog

- July 04, 2025: Initial setup with Streamlit crypto dashboard
- July 04, 2025: Integrated PostgreSQL database for persistent data storage

## User Preferences

Preferred communication style: Simple, everyday language.