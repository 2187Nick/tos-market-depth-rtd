# app.py
"""
Main application entry point for the bookmap style visualization.
This combines the RTD client with Dash visualization in a single process.
"""
import sys
import time
import threading
import webbrowser
import pythoncom
from threading import Timer
from queue import Queue
from queue import Empty

from config.quote_types import QuoteType
from src.core.error_handler import RTDError
from src.core.logger import get_logger
from src.core.settings import SETTINGS
from src.rtd.client import RTDClient
from dash_app import create_dash_app

# Set up logging
logger = get_logger(__name__)

# Global symbol queue for thread-safe symbol updates
symbol_queue = Queue()

def main():
    exchanges = ["A", "B", "C", "I", "N", "Q", "W", "X", "Z"]
    port = 8050
    
    logger.info("Starting application")
    
    # Initialize COM for the main thread
    pythoncom.CoInitialize()
    logger.info("COM initialized")
    
    try:
        # Initialize the RTD client first (must be in main thread)
        client = RTDClient(heartbeat_ms=SETTINGS['timing']['initial_heartbeat'], enable_db=True)
        client.initialize()  # Outside of context manager to keep it alive
        logger.info(f"RTD Client initialized with heartbeat: {client.heartbeat_interval}ms")
        
        # Create the Dash app but don't start it yet
        dash_app = create_dash_app("", update_interval=1)  # Start with empty symbol
        
        # Add RTD client to dash app for symbol subscriptions
        dash_app.rtd_client = client
        dash_app.exchanges = exchanges
        
        # Start a separate thread to launch browser and then start Dash
        def start_browser_and_dash():
            # Open browser first
            logger.info(f"Opening browser to http://127.0.0.1:{port}")
            webbrowser.open(f"http://127.0.0.1:{port}/")
            
            # Short delay to ensure browser is launched before Dash starts
            time.sleep(0.5)
            
            # Now start Dash - this will block this thread
            logger.info("Starting Dash server")
            dash_app.app.run(debug=False, port=port, host='127.0.0.1')
            
        # Start Dash in a daemon thread
        dash_thread = threading.Thread(
            target=start_browser_and_dash,
            daemon=True
        )
        dash_thread.start()
        logger.info(f"Dash app thread started")
        
        # Main RTD loop - keep processing in the main thread
        logger.info("Starting main RTD processing loop")
        
        # Print initial message
        print("\nApplication running - press Ctrl+C to exit")
        print(f"Visit http://127.0.0.1:{port}/ in your browser\n")
        
        try:
            while dash_thread.is_alive():
                # Process COM messages in main thread
                pythoncom.PumpWaitingMessages()
                
                # Short sleep to prevent high CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Application terminated by user")
            
    except RTDError as e:
        logger.error(f"RTD Error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1
    finally:
        try:
            # Clean up RTD client
            if 'client' in locals() and client:
                logger.info("Disconnecting RTD client")
                client.Disconnect()
                
            # Clean up COM in main thread
            pythoncom.CoUninitialize()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
        logger.info("Application shut down")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())