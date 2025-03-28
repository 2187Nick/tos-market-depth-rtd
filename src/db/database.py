"""
Database implementation for bookmap-style visualization of RTD data.

This module provides a simplified database structure optimized for
storing real-time market depth data for visualization.
"""
import os
import sqlite3
import time
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Tuple, Union, Any

from src.core.logger import get_logger
from src.core.settings import SETTINGS
from src.utils.quote import Quote


logger = get_logger(__name__)

class BookmapDatabase:
    """SQLite implementation for storing market depth data for bookmap visualization."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite database for RTD bookmap data.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default path.
        """
        self.logger = logger
        self.conn = None
        self._lock = Lock()
        
        # Use provided path or default from settings
        if db_path is None:
            storage_dir = SETTINGS.get('storage', {}).get('db_path', 'data')
            # Expand environment variables if present
            storage_dir = os.path.expandvars(storage_dir)
            # Create directory if it doesn't exist
            os.makedirs(storage_dir, exist_ok=True)
            
            # Create a daily database file
            today = datetime.now().strftime('%Y-%m-%d')
            db_path = os.path.join(storage_dir, f'bookmap_{today}.db')
        
        self.db_path = db_path
        self.logger.info(f"Bookmap database will be stored at: {self.db_path}")
    
    def initialize(self) -> bool:
        """
        Initialize the SQLite database and create necessary tables.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            with self._lock:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                # Enable WAL mode for better concurrency
                self.conn.execute('PRAGMA journal_mode=WAL')
                self.conn.execute('PRAGMA synchronous=NORMAL')
                self.conn.execute('PRAGMA temp_store=MEMORY')
                
                # Create market depth table to store aggregated sizes
                # This is our main table for bookmap visualization
                self.conn.execute('''
                CREATE TABLE IF NOT EXISTS market_depth (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,           -- Base symbol (.SPY250214C610)
                    price REAL NOT NULL,            -- Price level
                    bid_size INTEGER,               -- Aggregated bid size at this price
                    ask_size INTEGER,               -- Aggregated ask size at this price
                    timestamp REAL NOT NULL,        -- When this aggregation was calculated
                    UNIQUE(symbol, price, timestamp)
                )
                ''')
                
                # Create indexes for faster queries
                self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_market_depth_symbol 
                ON market_depth (symbol)
                ''')
                
                self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_market_depth_timestamp
                ON market_depth (timestamp)
                ''')
                
                # Create a table for latest quotes (current market state)
                self.conn.execute('''
                CREATE TABLE IF NOT EXISTS latest_quotes (
                    symbol TEXT NOT NULL,
                    quote_type TEXT NOT NULL,
                    value REAL,
                    timestamp REAL NOT NULL,
                    PRIMARY KEY (symbol, quote_type)
                )
                ''')
                
                # Create a table for last_size history to display as bubbles
                self.conn.execute('''
                CREATE TABLE IF NOT EXISTS last_size_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    size INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    UNIQUE(symbol, timestamp)
                )
                ''')
                
                # Index for last_size_history
                self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_last_size_history_symbol_ts
                ON last_size_history (symbol, timestamp)
                ''')
                
                self.conn.commit()
                self.logger.info("Bookmap database initialized successfully")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            return False
            
        except Exception as e:
            self.logger.error(f"Unexpected error during database initialization: {str(e)}")
            return False
    
    def store_quotes(self, quotes: List[Quote]) -> int:
        """
        Process quotes and update market depth data.
        
        Args:
            quotes: List of Quote objects to process
            
        Returns:
            int: Number of depth updates stored
        """
        if not quotes:
            return 0
            
        try:
            with self._lock:
                # Update latest quotes table with all incoming quotes
                latest_data = []
                
                # Track quotes needed for market depth updates
                depth_quotes = []
                symbols = set()
                
                # Track LAST_SIZE quotes
                last_size_data = []
                
                for quote in quotes:
                    quote_type_str = quote.quote_type.value if hasattr(quote.quote_type, 'value') else str(quote.quote_type)
                    
                    # Update latest quotes
                    latest_data.append((
                        quote.symbol,
                        quote_type_str,
                        quote.value,
                        quote.timestamp
                    ))
                    
                    # Special handling for LAST_SIZE quotes
                    if quote_type_str == 'LAST_SIZE':  # Removed size filter
                        # Get the LAST price for this symbol and timestamp
                        cursor = self.conn.execute('''
                        SELECT value FROM latest_quotes
                        WHERE symbol = ? AND quote_type = 'LAST'
                        ORDER BY timestamp DESC LIMIT 1
                        ''', (quote.symbol,))
                        
                        last_price_row = cursor.fetchone()
                        if last_price_row:
                            last_price = last_price_row[0]
                            # Add to last_size_history data
                            last_size_data.append((
                                quote.symbol,
                                last_price,
                                int(quote.value),  # Make sure size is an integer
                                quote.timestamp
                            ))
                    
                    # Track quotes needed for market depth updates
                    if quote_type_str in ['BID', 'ASK', 'BID_SIZE', 'ASK_SIZE']:
                        depth_quotes.append(quote)
                        # Extract base symbol (without exchange identifier)
                        base_symbol = quote.symbol.split('&')[0] if '&' in quote.symbol else quote.symbol
                        symbols.add(base_symbol)
                
                # Update latest quotes table
                if latest_data:
                    self.conn.executemany('''
                    INSERT OR REPLACE INTO latest_quotes
                    (symbol, quote_type, value, timestamp)
                    VALUES (?, ?, ?, ?)
                    ''', latest_data)
                
                # Update last_size_history table
                if last_size_data:
                    self.conn.executemany('''
                    INSERT OR REPLACE INTO last_size_history
                    (symbol, price, size, timestamp)
                    VALUES (?, ?, ?, ?)
                    ''', last_size_data)
                
                # Process market depth for each symbol
                depth_updates = 0
                for symbol in symbols:
                    updates = self._update_market_depth_for_symbol(symbol)
                    depth_updates += updates
                
                self.conn.commit()
                return depth_updates
                
        except sqlite3.Error as e:
            self.logger.error(f"Error storing quotes: {str(e)}")
            return 0
            
        except Exception as e:
            self.logger.error(f"Unexpected error storing quotes: {str(e)}")
            return 0
    
    def _update_market_depth_for_symbol(self, base_symbol: str) -> int:
        """
        Update market depth for a specific symbol by aggregating across exchanges.
        
        Args:
            base_symbol: Base symbol without exchange identifier
            
        Returns:
            int: Number of price levels updated
        """
        try:
            # Get all related symbols (base + all exchanges)
            cursor = self.conn.execute('''
            SELECT DISTINCT symbol FROM latest_quotes
            WHERE symbol = ? OR symbol LIKE ? || '&%'
            ''', (base_symbol, base_symbol))
            
            symbols = [row[0] for row in cursor.fetchall()]
            if not symbols:
                return 0
            
            # Collect BID/ASK prices for each symbol
            prices = {}  # {symbol: {'BID': price, 'ASK': price}}
            for symbol in symbols:
                cursor = self.conn.execute('''
                SELECT quote_type, value FROM latest_quotes
                WHERE symbol = ? AND quote_type IN ('BID', 'ASK')
                ''', (symbol,))
                
                symbol_prices = {}
                for row in cursor.fetchall():
                    symbol_prices[row[0]] = row[1]
                    
                if symbol_prices:
                    prices[symbol] = symbol_prices
            
            # Collect sizes for each price point
            depth_data = {}  # {price: {'bid_size': X, 'ask_size': Y}}
            timestamp = time.time()
            
            for symbol, price_data in prices.items():
                # Get BID_SIZE at this price
                if 'BID' in price_data and price_data['BID'] is not None:
                    bid_price = price_data['BID']
                    cursor = self.conn.execute('''
                    SELECT value FROM latest_quotes
                    WHERE symbol = ? AND quote_type = 'BID_SIZE'
                    ''', (symbol,))
                    
                    row = cursor.fetchone()
                    if row and row[0] is not None:
                        if bid_price not in depth_data:
                            depth_data[bid_price] = {'bid_size': 0, 'ask_size': 0}
                        depth_data[bid_price]['bid_size'] += row[0]
                
                # Get ASK_SIZE at this price
                if 'ASK' in price_data and price_data['ASK'] is not None:
                    ask_price = price_data['ASK']
                    cursor = self.conn.execute('''
                    SELECT value FROM latest_quotes
                    WHERE symbol = ? AND quote_type = 'ASK_SIZE'
                    ''', (symbol,))
                    
                    row = cursor.fetchone()
                    if row and row[0] is not None:
                        if ask_price not in depth_data:
                            depth_data[ask_price] = {'bid_size': 0, 'ask_size': 0}
                        depth_data[ask_price]['ask_size'] += row[0]
            
            # Store aggregated depth data
            for price, sizes in depth_data.items():
                self.conn.execute('''
                INSERT OR REPLACE INTO market_depth 
                (symbol, price, bid_size, ask_size, timestamp)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    base_symbol,
                    price,
                    sizes['bid_size'],
                    sizes['ask_size'],
                    timestamp
                ))
            
            return len(depth_data)
                
        except Exception as e:
            self.logger.error(f"Error updating market depth: {str(e)}")
            return 0

    def get_market_depth_history(self, symbol: str, 
                               start_time: Optional[float] = None,
                               end_time: Optional[float] = None) -> List[Dict]:
        """
        Get market depth history for visualization.
        
        Args:
            symbol: Symbol to retrieve depth for
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            list: List of market depth records
        """
        try:
            with self._lock:
                query = '''
                SELECT price, bid_size, ask_size, timestamp
                FROM market_depth
                WHERE symbol = ?
                '''
                params = [symbol]
                
                if start_time is not None:
                    query += ' AND timestamp >= ?'
                    params.append(start_time)
                    
                if end_time is not None:
                    query += ' AND timestamp <= ?'
                    params.append(end_time)
                    
                query += ' ORDER BY timestamp ASC, price DESC'
                
                cursor = self.conn.execute(query, params)
                results = cursor.fetchall()
                
                return [{
                    'price': row[0],
                    'bid_size': row[1],
                    'ask_size': row[2],
                    'timestamp': row[3]
                } for row in results]
                
        except Exception as e:
            self.logger.error(f"Error retrieving market depth history: {str(e)}")
            return []
    
    def get_price_levels(self, symbol: str, 
                       start_time: Optional[float] = None, 
                       end_time: Optional[float] = None) -> List[float]:
        """
        Get all unique price levels for a symbol in the given time range.
        Useful for establishing the y-axis of the bookmap.
        
        Args:
            symbol: Symbol to retrieve prices for
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            list: List of unique price levels sorted in descending order
        """
        try:
            with self._lock:
                query = '''
                SELECT DISTINCT price 
                FROM market_depth
                WHERE symbol = ?
                '''
                params = [symbol]
                
                if start_time is not None:
                    query += ' AND timestamp >= ?'
                    params.append(start_time)
                    
                if end_time is not None:
                    query += ' AND timestamp <= ?'
                    params.append(end_time)
                    
                query += ' ORDER BY price DESC'
                
                cursor = self.conn.execute(query, params)
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Error retrieving price levels: {str(e)}")
            return []
    
    def get_time_slices(self, symbol: str,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None) -> List[float]:
        """
        Get all unique timestamps for a symbol in the given time range.
        Useful for establishing the x-axis of the bookmap.
        
        Args:
            symbol: Symbol to retrieve timestamps for
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            list: List of unique timestamps in ascending order
        """
        try:
            with self._lock:
                query = '''
                SELECT DISTINCT timestamp 
                FROM market_depth
                WHERE symbol = ?
                '''
                params = [symbol]
                
                if start_time is not None:
                    query += ' AND timestamp >= ?'
                    params.append(start_time)
                    
                if end_time is not None:
                    query += ' AND timestamp <= ?'
                    params.append(end_time)
                    
                query += ' ORDER BY timestamp ASC'
                
                cursor = self.conn.execute(query, params)
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Error retrieving time slices: {str(e)}")
            return []
    
    def close(self) -> None:
        """Close the database connection."""
        try:
            with self._lock:
                if self.conn:
                    self.conn.close()
                    self.logger.info("Database connection closed")
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error closing database: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error closing database: {str(e)}")


# Factory function to create database instance
def create_database(**kwargs) -> BookmapDatabase:
    """
    Create a database instance for bookmap visualization.
    
    Args:
        **kwargs: Additional arguments for database constructor
        
    Returns:
        BookmapDatabase: Database instance
    """
    return BookmapDatabase(**kwargs)