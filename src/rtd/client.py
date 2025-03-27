from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import pythoncom
import time
import threading
from queue import Queue, Empty

from comtypes import COMObject, GUID
from comtypes.automation import VARIANT, VARIANT_BOOL
from comtypes.client import CreateObject

from config.quote_types import QuoteType
from src.core.error_handler import (
    RTDClientError,
    RTDConnectionError,
    RTDConnectionState,
    RTDHeartbeatError,
    RTDServerError,
    RTDUpdateError,
    handle_com_error,
    log_method_call,
    validate_connection_state
)
from src.core.logger import get_logger
from src.core.settings import SETTINGS
from src.rtd.interfaces import IRTDUpdateEvent, IRtdServer
from src.utils import cleanup, state, topic
from src.utils.quote import Quote
from src.db.database import create_database


class RTDClient(COMObject):
    """
    Real-Time Data Client for ThinkorSwim RTD Server.
    
    This class provides a synchronous interface to the ThinkorSwim RTD Server,
    handling real-time market data subscriptions and updates.
    
    Attributes:
        _state (RTDConnectionState): Current connection state
        server (IRtdServer): COM server instance
        topics (Dict[int, Tuple[str, str]]): Active topic subscriptions
        heartbeat_interval (int): Server heartbeat interval in milliseconds
    """
    _com_interfaces_ = [IRTDUpdateEvent]

    def __init__(
        self, 
        heartbeat_ms: Optional[int] = None,
        logger: Optional[Any] = None,
        enable_db: Optional[bool] = None
    ) -> None:
        """
        Initialize the RTD Client.

        Args:
            heartbeat_ms: Optional heartbeat interval in milliseconds.
                         Defaults to value from config.
            logger: Optional logger instance. If None, creates a new logger.
            enable_db: Whether to enable database storage. If None, uses config.

        Raises:
            RTDClientError: If initialization fails
        """
        super().__init__()
        print("RTDClient initialization started")
        
        # Initialize logger
        self.logger = logger or get_logger("RTDClient")
        
        # COM server and state
        self.server: Optional[IRtdServer] = None
        self._state = RTDConnectionState.DISCONNECTED
        self._lock = Lock()
        
        # Topic management
        self.topics: Dict[int, Tuple[str, str]] = {}
        self._topic_lock = Lock()
        self._latest_values: Dict[Tuple[str, str], Quote] = {} 
        self._value_lock = Lock() 
        
        # Heartbeat configuration
        self._heartbeat_interval = (
            heartbeat_ms or 
            SETTINGS['timing']['initial_heartbeat']
        )
        
        # Update tracking
        self._update_notify_count = 0
        self._last_refresh_time = None
        
        self.logger.info("RTD Client instance created")

        # Database configuration
        self._enable_db = enable_db if enable_db is not None else SETTINGS.get('database', {}).get('auto_capture', True)
        self._db = None
        self._quote_queue = Queue()
        self._db_worker = None
        self._worker_running = False
        self._debug_counter = 0  # Counter for debugging
        
        if self._enable_db:
            self._init_database()
            
        print("RTDClient __init__ completed")

    def __enter__(self) -> 'RTDClient':
        """
        Enter the runtime context for using the RTD client.
        
        Initializes the COM server and establishes the connection.
        
        Returns:
            RTDClient: Self reference for context manager use
            
        Raises:
            RTDServerError: If server initialization fails
        """
        self.initialize()
        return self
    
    def _init_database(self) -> None:
        """Initialize the database for bookmap visualization."""
        try:
            self.logger.info("Initializing bookmap database for market depth visualization")
            print("Initializing database")
            self._db = create_database()
            
            if not self._db.initialize():
                self.logger.error("Failed to initialize bookmap database. Data capture disabled.")
                self._db = None
                return
                
            # Set worker running flag before starting thread
            self._worker_running = True
            self._debug_counter = 0
            
            # Start worker thread for processing quotes - with a shorter timeout
            self._db_worker = threading.Thread(
                target=self._process_quote_queue,
                daemon=True,
                name="BookmapDBWorkerThread"
            )
            self._db_worker.start()
            
            # Wait briefly to ensure thread starts
            time.sleep(0.1)
            
            # Verify thread is running
            if not self._db_worker.is_alive():
                self.logger.error("Database worker thread failed to start")
                self._worker_running = False
                return
                
            self.logger.info("Bookmap database worker thread started")
            print("Database worker thread started")
            
        except Exception as e:
            self.logger.error(f"Error setting up bookmap database: {str(e)}")
            print(f"Database init error: {str(e)}")
            self._worker_running = False
            self._db = None

    # Processing quotes from the queue and store in database
    def _process_quote_queue(self) -> None:
        """Worker thread to process quotes from the queue and store in database."""
        print("Database worker thread running")
        
        # Use smaller batch size and shorter interval to be more responsive
        batch_size = SETTINGS.get('database', {}).get('batch_size', 100)
        batch_interval = .1 #1.0  # Use shorter timeout of 1 second (instead of 5)
        
        self.logger.info(f"Quote processor configured with batch size {batch_size}, interval {batch_interval}s")
        queue_empty_count = 0
        
        while self._worker_running:
            try:
                quotes = []
                
                # Get quotes with a short timeout - non-blocking approach
                try:
                    # Try to get a quote with a short timeout
                    quote = self._quote_queue.get(block=True, timeout=batch_interval)
                    quotes.append(quote)
                    self._quote_queue.task_done()
                    
                    # Get any additional immediately available quotes
                    for _ in range(batch_size - 1):
                        try:
                            quote = self._quote_queue.get(block=False)
                            quotes.append(quote)
                            self._quote_queue.task_done()
                        except Empty:
                            break
                    
                    queue_empty_count = 0  # Reset counter when we get data
                    
                except Empty:
                    queue_empty_count += 1
                    if queue_empty_count % 30 == 0:  # Report every 30 seconds with 1s timeout
                        self.logger.debug(f"Queue has been empty for ~{queue_empty_count} intervals")
                        self.logger.debug(f"Worker still running. Topics: {len(self.topics)}, Updates: {self._update_notify_count}")
                    continue  # Continue the loop instead of potentially falling through
                except Exception as e:
                    self.logger.error(f"Error getting from queue: {str(e)}")
                    time.sleep(0.1)  # Brief pause on error
                    continue  # Continue the loop instead of potentially falling through
                    
                # Store quotes if we have any
                if quotes and self._db:
                    try:
                        self._db.store_quotes(quotes)
                        if self._debug_counter < 5:  # Log first few successful stores
                            self.logger.info(f"Successfully stored {len(quotes)} quotes")
                            self._debug_counter += 1
                    except Exception as e:
                        self.logger.error(f"Error storing quotes: {str(e)}")
                        time.sleep(0.1)  # Brief pause on error
                
            except Exception as e:
                self.logger.error(f"Error in quote processor: {str(e)}")
                time.sleep(0.1)  # Short sleep on error to prevent tight loop
                continue  # Continue the loop instead of potentially falling through
                
        self.logger.info("Quote processor thread exiting")
        print("Quote processor thread exiting")

    @handle_com_error(RTDServerError)
    @log_method_call()
    def initialize(self) -> None:
        """
        Initialize the RTD server connection.
        
        Performs COM initialization and server startup sequence.
        Should be called before any other operations.
        
        Raises:
            RTDServerError: If server initialization fails
            RTDConnectionError: If called in invalid state
        """
        if self._state != RTDConnectionState.DISCONNECTED:
            raise RTDConnectionError(
                f"Initialization attempted in invalid state: {self._state}"
            )
            
        self._state = RTDConnectionState.CONNECTING
        self.logger.info("Starting RTD server initialization")
        print("Starting RTD server initialization")
        
        try:
            # Initialize COM for the current thread
            print("client.py about to initialize COM")
            pythoncom.CoInitialize()
            print("COM initialized")

            time.sleep(1)  # Small delay to ensure COM is ready
            
            # Create COM server instance
            self.server = CreateObject(
                GUID(SETTINGS['rtd']['progid']), 
                interface=IRtdServer
            )
            self.logger.info("COM server instance created")
            print("COM server instance created")
            
            # Start the server
            result = self.server.ServerStart(self)
            
            if result == 1:
                self._state = RTDConnectionState.CONNECTED
                self.logger.info("Server started successfully")
                print("RTD Server started successfully")
                
                # Configure heartbeat
                current_interval = self.heartbeat_interval
                self.heartbeat_interval = SETTINGS['timing']['default_heartbeat']
                self.logger.info(
                    f"Heartbeat interval updated: {current_interval}ms -> "
                    f"{self.heartbeat_interval}ms"
                )
            else:
                raise RTDServerError(f"ServerStart failed with result: {result}")
                
        except Exception as e:
            self._state = RTDConnectionState.DISCONNECTED
            self.logger.error(f"Server initialization failed: {str(e)}")
            print(f"Server initialization failed: {str(e)}")
            cleanup.cleanup_com() 
            raise

    # The rest of the methods remain unchanged
    @handle_com_error(RTDClientError)
    @log_method_call()
    @validate_connection_state([RTDConnectionState.CONNECTED])
    def subscribe(self, quote_type: Union[str, QuoteType], symbol: str) -> Optional[int]:
        with self._topic_lock:

            quote_type_str = topic.validate_quote_type(quote_type)
            topic_id = topic.generate_topic_id(quote_type_str, symbol)
            
            if topic_id in self.topics:
                self.logger.info(
                    f"Already subscribed to {symbol} {quote_type_str}"
                )
                return topic_id
                
            # subscription params per current specs
            strings = (VARIANT * 2)()
            strings[0].value = quote_type_str
            strings[1].value = symbol
            get_new_values = VARIANT_BOOL(True)
            
            try:
                result = self.server.ConnectData(
                    topic_id, strings, get_new_values
                )
                self.logger.debug(f"Subscription raw result {result}")
                
                if isinstance(result, list) and len(result) >= 1 and result[0]:
                    self.topics[topic_id] = (symbol, quote_type_str)
                    self.logger.debug(
                        f"Subscribed to {symbol} {quote_type_str} "
                        f"with ID {topic_id}"
                    )
                    return topic_id
                else:
                    self.logger.warning(
                        f"Subscription failed for {symbol} {quote_type_str}"
                    )
                    return None
                    
            except Exception as e:
                self.logger.error(
                    f"Error subscribing to {symbol} {quote_type_str}: {e}"
                )
                raise RTDClientError(
                    f"Subscription failed for {symbol}"
                ) from e

    @handle_com_error(RTDClientError)
    @log_method_call()
    @validate_connection_state([RTDConnectionState.CONNECTED, RTDConnectionState.DISCONNECTING])
    def unsubscribe(self, quote_type: Union[str, QuoteType], symbol: str) -> bool:
        with self._topic_lock:
            quote_type_str = topic.validate_quote_type(quote_type)
            
            topic_id = topic.find_topic_id(self.topics, symbol, quote_type_str)
            if topic_id is None:
                self.logger.warning(
                    f"Not subscribed to {symbol} {quote_type_str}"
                )
                return False
                
            try:
                result = self.server.DisconnectData(topic_id)
                self.logger.debug(f"Unsub raw result {result}")
                
                if result == 0:  # Success
                    del self.topics[topic_id]
                    self.logger.debug(
                        f"Unsubscribed from {symbol} {quote_type_str}"
                    )
                    return True
                else:
                    self.logger.warning(
                        f"Unsubscription failed for {symbol} {quote_type_str}"
                    )
                    return False
                    
            except Exception as e:
                self.logger.error(
                    f"Error unsubscribing from {symbol} {quote_type_str}: {e}"
                )
                return False


    @handle_com_error(RTDUpdateError)
    @log_method_call()
    @validate_connection_state([RTDConnectionState.CONNECTED])
    def UpdateNotify(self) -> bool:
        self._update_notify_count += 1
        # Only log occasionally to reduce noise
        if self._update_notify_count == 1 or self._update_notify_count % 100 == 0:
            self.logger.debug(f"UpdateNotify called (count: {self._update_notify_count})")
            print(f"Update notification #{self._update_notify_count}")
        return self.refresh_topics()

    @handle_com_error(RTDClientError)
    @log_method_call()
    @validate_connection_state([RTDConnectionState.CONNECTED])
    def refresh_topics(self) -> bool:
        try:
            result = self.server.RefreshData()
            self._last_refresh_time = time.time()
            
            if not result or not isinstance(result, list) or len(result) != 2:
                self.logger.warning(f"Unexpected result format from RefreshData: {result}")
                return False

            topic_count, data = result
            
            # Only log meaningful updates
            if topic_count > 0 and self._update_notify_count % 100 == 0:
                print(f"Refresh data for {topic_count} topics")
                
            if topic_count == 0 or not data:
                return True

            if isinstance(data, tuple) and len(data) == 2:
                topic_ids, raw_values = data
                for id, raw_value in zip(topic_ids, raw_values):
                    if id in self.topics:
                        symbol, quote_type = self.topics[id]
                        quote_obj = Quote(quote_type, symbol, raw_value)
                        self._handle_quote_update(id, symbol, quote_type, quote_obj)
                return True
            else:
                self.logger.warning(f"Unexpected data format in RefreshData result: {data}")
                return False

        except Exception as e:
            self.logger.error(f"Error fetching or processing refresh data: {e}", exc_info=True)
            print(f"ERROR in refresh_topics: {str(e)}")
            return False


    def _handle_quote_update(self, id: int, symbol: str, quote_type: str, quote: Quote) -> None:
        try:
            if quote.value is None:
                self.logger.debug(f"Null value received for {symbol} {quote_type}")
                return

            # Update latest value
            with self._value_lock:
                key = (symbol, quote_type)
                old_value = None
                if key in self._latest_values:
                    old_value = self._latest_values[key].value
                self._latest_values[key] = quote
                value_changed = old_value != quote.value

            if value_changed:
                # Log first quote or occasional quotes for debugging
                if self._update_notify_count < 5 or self._update_notify_count % 500 == 0:
                    print(f"Quote: {symbol} {quote_type}: {quote.value}")
                
                # Add to database queue if it's a market depth quote type or LAST/LAST_SIZE
                if self._db and self._worker_running and quote_type in ['BID', 'ASK', 'BID_SIZE', 'ASK_SIZE', 'LAST', 'LAST_SIZE']:
                    self._quote_queue.put(quote)
                    # Log first few quotes being added to the queue
                    if self._update_notify_count < 20 or (quote_type in ['LAST', 'LAST_SIZE'] and self._update_notify_count % 100 == 0):
                        print(f"Added to queue: {symbol} {quote_type}: {quote.value}")

                    # For LAST quotes, add a special market depth entry with both bid and ask = 0
                    # This way we can distinguish LAST price points from real market depth
                    if quote_type == 'LAST' and self._db and self._worker_running:
                        last_price = quote.value
                        timestamp = quote.timestamp
                        
                        # Create a special LAST quote entry with size=0 to mark it as a LAST point
                        bid_quote = Quote('BID', symbol, last_price, timestamp)
                        ask_quote = Quote('ASK', symbol, last_price, timestamp)
                        bid_size_quote = Quote('BID_SIZE', symbol, 0, timestamp)  # Use 0 size to distinguish from real depth
                        ask_size_quote = Quote('ASK_SIZE', symbol, 0, timestamp)  # Use 0 size to distinguish from real depth
                        
                        # Add these quotes to the queue
                        self._quote_queue.put(bid_quote)
                        self._quote_queue.put(ask_quote)
                        self._quote_queue.put(bid_size_quote)
                        self._quote_queue.put(ask_size_quote)
                        
                        if self._update_notify_count < 20 or self._update_notify_count % 100 == 0:
                            print(f"Added LAST price {last_price} as market depth points")

        except Exception as e:
            self.logger.error(f"Error handling quote update: {e}")
            print(f"Error in _handle_quote_update: {str(e)}")

    @handle_com_error(RTDHeartbeatError)
    @log_method_call()
    @validate_connection_state([RTDConnectionState.CONNECTED, RTDConnectionState.DISCONNECTED])
    def check_heartbeat(self) -> bool:
        if self._state == RTDConnectionState.DISCONNECTED:
            self.logger.debug("Heartbeat check skipped - disconnected state")
            return False
            
        try:
            result = self.server.Heartbeat()
            is_healthy = result == 1
            
            if not is_healthy:
                self.logger.warning(
                    f"Unhealthy heartbeat response: {result}"
                )
            
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"Heartbeat check failed: {e}")
            raise RTDHeartbeatError("Heartbeat operation failed") from e

    @property
    def heartbeat_interval(self) -> int:
        return self._heartbeat_interval

    @heartbeat_interval.setter
    def heartbeat_interval(self, interval: int) -> None:
        if interval <= 0:
            raise ValueError("Heartbeat interval must be positive")
            
        self._heartbeat_interval = interval
        self.logger.info(f"Heartbeat interval set to {interval}ms")

    @handle_com_error(RTDServerError)
    @log_method_call()
    @validate_connection_state([RTDConnectionState.CONNECTED, RTDConnectionState.CONNECTING])
    def Disconnect(self) -> None:
        with self._lock:
            if self._state == RTDConnectionState.DISCONNECTED:
                self.logger.info("Already disconnected")
                return
                
            if self._state == RTDConnectionState.DISCONNECTING:
                self.logger.info("Disconnect already in progress")
                return
                
            self._state = RTDConnectionState.DISCONNECTING
            self.logger.info("Starting disconnect sequence")
            
            try:

                # Stop database worker
                if self._worker_running:
                    self.logger.info("Stopping database worker thread")
                    self._worker_running = False
                    if self._db_worker and self._db_worker.is_alive():
                        self._db_worker.join(timeout=5.0)
                        
                # Close database connection
                if self._db:
                    self.logger.info("Closing database connection")
                    self._db.close()
                    self._db = None
                    
                # Unsubscribe but can be optional as Excel doesn't seem to do it or not
                # very effectively for large number of topics
                subscriptions = [(qt, sym) for sym, qt in self.topics.values()]
                if subscriptions:
                    unsubscribe_results = self.batch_unsubscribe(subscriptions)
                    
                # Clear any remaining topics from memory
                cleanup.cleanup_topics(self.topics)
                
                if self.server is not None:
                    try:
                        self.server.ServerTerminate()
                        self.logger.info("Server terminated")
                    except Exception as e:
                        self.logger.error(f"Error terminating server: {e}")
                    finally:
                        self.server = None
                
                cleanup.cleanup_com()
                self._state = RTDConnectionState.DISCONNECTED
                self.logger.info("Disconnect completed")
                
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
                raise

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> None:
        try:
            if exc_type is not None:
                self.logger.error(f"Context exit due to error: {exc_val}")
            self.Disconnect()
        except Exception as e:
            self.logger.error(f"Error during context exit: {e}")
            if exc_type is None:
                raise

################################################
################ Helpers #######################
################################################

    def batch_subscribe(
        self,
        subscriptions: List[Tuple[Union[str, QuoteType], str]]
    ) -> Dict[Tuple[str, str], bool]:
        results = {}
        for quote_type, symbol in subscriptions:
            try:
                topic_id = self.subscribe(quote_type, symbol)
                results[(str(quote_type), symbol)] = topic_id is not None
            except Exception as e:
                self.logger.error(
                    f"Error in batch subscribe for {symbol} {quote_type}: {e}"
                )
                results[(str(quote_type), symbol)] = False
                
        successful = sum(1 for result in results.values() if result)
        self.logger.info(
            f"Batch subscribe completed: {successful}/{len(subscriptions)} "
            "successful"
        )
        return results

    def batch_unsubscribe(
        self,
        subscriptions: List[Tuple[Union[str, QuoteType], str]]
    ) -> Dict[Tuple[str, str], bool]:
        results = {}
        for quote_type, symbol in subscriptions:
            try:
                success = self.unsubscribe(quote_type, symbol)
                results[(str(quote_type), symbol)] = success
            except Exception as e:
                self.logger.error(
                    f"Error in batch unsubscribe for {symbol} {quote_type}: {e}"
                )
                results[(str(quote_type), symbol)] = False
                
        successful = sum(1 for result in results.values() if result)
        self.logger.info(
            f"Batch unsubscribe completed: {successful}/{len(subscriptions)} "
            "successful"
        )
        return results

    def change_symbol(self, new_symbol: str, exchanges: List[str] = None) -> bool:
        """
        Change the current symbol, unsubscribing from old topics and subscribing to new ones.
        
        Args:
            new_symbol: The new symbol to subscribe to
            exchanges: Optional list of exchanges to subscribe to. If None, uses default exchanges.
            
        Returns:
            bool: True if symbol change was successful, False otherwise
        """
        try:
            self.logger.info(f"Changing symbol to: {new_symbol}")
            
            # First unsubscribe from all existing topics
            if self.topics:
                self.logger.info(f"Unsubscribing from {len(self.topics)} existing topics")
                for topic_id, (old_symbol, quote_type) in self.topics.copy().items():
                    self.unsubscribe(quote_type, old_symbol)
                    self.logger.info(f"Unsubscribed from {old_symbol} {quote_type}")
            
            # Define quote types for base symbol
            quote_types = [QuoteType.BID, QuoteType.ASK, QuoteType.BID_SIZE, QuoteType.ASK_SIZE, QuoteType.LAST, QuoteType.LAST_SIZE]
            
            # Subscribe to base symbol
            self.logger.info(f"Subscribing to base symbol: {new_symbol}")
            for quote_type in quote_types:
                success = self.subscribe(quote_type, new_symbol)
                self.logger.info(f"Subscription attempt for {new_symbol} {quote_type.name}: {'Success' if success else 'Failed'}")
            
            # Subscribe to exchange-specific symbols if exchanges provided
            if exchanges:
                exchange_quote_types = [QuoteType.BID, QuoteType.ASK, QuoteType.BID_SIZE, QuoteType.ASK_SIZE]
                for exchange in exchanges:
                    symbol_ex = f"{new_symbol}&{exchange}"
                    for quote_type in exchange_quote_types:
                        if self.subscribe(quote_type, symbol_ex):
                            self.logger.info(f"Subscribed to {symbol_ex} {quote_type.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error changing symbol: {e}")
            return False

    def __str__(self) -> str:
        #status = "Connected" if self._state == RTDConnectionState.CONNECTED else "Disconnected"
        status = "Connected" if self.is_connected else "Disconnected"
        topic_count = len(self.topics)
        return (
            f"RTDClient: {status}, "
            f"Topics: {topic_count}, "
            f"Updates: {self._update_notify_count}"
        )

    def __repr__(self) -> str:
        return (
            f"RTDClient(state={self._state.name}, "
            f"topics={len(self.topics)}, "
            f"heartbeat={self._heartbeat_interval}ms, "
            f"updates={self._update_notify_count})"
        )