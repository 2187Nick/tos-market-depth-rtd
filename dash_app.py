# dashapp.py
from dash import Dash, Input, Output, State, html, dcc, exceptions, ClientsideFunction, no_update
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import time
import threading
import webbrowser
from threading import Timer
import os.path
import json
import pandas as pd

from src.db.database import create_database
from src.utils import bookmap_viz
from src.core.logger import get_logger

logger = get_logger(__name__)

class BookmapDashApp:
    def __init__(self, symbol: str, update_interval: int = 2):
        # Create app with custom stylesheet
        self.app = Dash(
            __name__, 
            # Add a simple CSS to make background black
            external_stylesheets=[
                {
                    'href': 'https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap',
                    'rel': 'stylesheet'
                }
            ],
            suppress_callback_exceptions=True  # Add this to handle dynamic callbacks better
        )
        
        # Add custom CSS to make the body background black and remove margins
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    body {
                        margin: 0;
                        padding: 0;
                        background-color: black;
                        overflow: hidden;
                    }
                    #react-entry-point {
                        height: 100vh;
                    }
                    .control-panel {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        gap: 10px;
                        padding: 5px;
                    }
                    .control-button {
                        background-color: #333;
                        color: white;
                        border: 1px solid #555;
                        padding: 5px 15px;
                        cursor: pointer;
                        border-radius: 4px;
                    }
                    .control-button:hover {
                        background-color: #444;
                    }
                    .symbol-input {
                        background-color: #333;
                        color: white;
                        border: 1px solid #555;
                        padding: 5px;
                        border-radius: 4px;
                    }
                    .start-button {
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        cursor: pointer;
                        border-radius: 4px;
                        font-size: 16px;
                    }
                    .start-button:hover {
                        background-color: #45a049;
                    }
                </style>
                <script>
                    // Function to get stored symbol
                    function getStoredSymbol() {
                        const symbol = localStorage.getItem('lastSymbol');
                        console.log('Retrieved stored symbol:', symbol);
                        return symbol;
                    }
                    
                    // Function to set stored symbol
                    function setStoredSymbol(symbol) {
                        console.log('Storing symbol:', symbol);
                        localStorage.setItem('lastSymbol', symbol);
                    }
                    
                    // Initialize on page load
                    window.addEventListener('load', function() {
                        console.log('Page loaded, checking for stored symbol');
                        const storedSymbol = getStoredSymbol();
                        if (storedSymbol) {
                            console.log('Found stored symbol:', storedSymbol);
                            const symbolInput = document.querySelector('#symbol-input');
                            if (symbolInput) {
                                symbolInput.value = storedSymbol;
                                console.log('Set input value to stored symbol');
                            }
                        }
                    });
                </script>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        self.symbol = symbol
        self.update_interval = update_interval
        self.db = None
        self.init_database()
        
        # Create an empty bookmap figure to show initially
        empty_fig = bookmap_viz.create_bookmap_figure(title="Enter a symbol to start")
        empty_fig.add_annotation(
            text="Enter a symbol and click Start to begin",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=24, color="white")
        )
        
        # Set up application layout with full screen styling
        self.app.layout = html.Div([
            html.H1(id='title-header',
                   children=f"Live Market Depth",
                   style={'color': 'white', 'textAlign': 'center', 'margin': '5px 0'}),
            # Add control panel
            html.Div([
                dcc.Input(
                    id='symbol-input',
                    type='text',
                    value="",
                    className='symbol-input',
                    placeholder='Enter symbol...'
                ),
                html.Button(
                    'Start',
                    id='start-button',
                    className='start-button'
                ),
                html.Button(
                    'Pause',
                    id='pause-button',
                    className='control-button',
                    style={'display': 'none'}
                )
            ], className='control-panel'),
            html.Div(id='status-message',
                    style={'color': 'yellow', 'textAlign': 'center', 'margin': '0 0 5px 0'}),
            dcc.Graph(id='live-bookmap',
                     figure=empty_fig,
                     style={'height': '88vh', 'width': '100%'},
                     config={'displayModeBar': True, 'responsive': True}),
            dcc.Store(id='zoom-store', storage_type='memory'),
            dcc.Store(id='app-state', storage_type='memory', data={'is_paused': True, 'is_started': False}),
            dcc.Store(id='symbol-store', storage_type='memory', data={'symbol': ""}),
            dcc.Store(id='store-symbol-clientside', storage_type='memory'),
            dcc.Interval(
                id='interval-component',
                interval=update_interval * 1000,
                n_intervals=0,
                disabled=True
            )
        ])
        
        self.setup_callbacks()
        
    def init_database(self):
        """Initialize database connection with retries"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.db = create_database()
                if self.db.initialize():
                    logger.info("Database initialized successfully")
                    # Verify database has market_depth table
                    cursor = self.db.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_depth'")
                    if cursor.fetchone() is None:
                        logger.error("Database doesn't contain market_depth table")
                        time.sleep(1)
                        continue
                    return True
            except Exception as e:
                logger.error(f"Database initialization attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        
        raise RuntimeError("Failed to initialize database after multiple attempts")
    
    def update_symbol(self, new_symbol: str) -> None:
        """
        Update the symbol being displayed.
        
        Args:
            new_symbol: The new symbol to display
        """
        logger.info(f"Updating display symbol from {self.symbol} to {new_symbol}")
        
        # Try to subscribe to the new symbol
        if hasattr(self, 'rtd_client') and hasattr(self, 'exchanges'):
            success = self.rtd_client.change_symbol(new_symbol, self.exchanges)
            if not success:
                logger.error(f"Failed to subscribe to symbol: {new_symbol}")
                raise Exception(f"Failed to subscribe to symbol: {new_symbol}")
        
        self.symbol = new_symbol
        return {'symbol': new_symbol}
    
    def setup_callbacks(self):
        # Add clientside callback to store symbol in localStorage
        app = self.app
        
        app.clientside_callback(
            """
            function(symbol_data) {
                if (symbol_data && symbol_data.symbol) {
                    console.log('Storing symbol in localStorage:', symbol_data.symbol);
                    localStorage.setItem('lastSymbol', symbol_data.symbol);
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output('store-symbol-clientside', 'data'),
            Input('symbol-store', 'data')
        )

        # Callback to update the title when symbol changes
        @self.app.callback(
            Output('title-header', 'children'),
            [Input('symbol-store', 'data')]
        )
        def update_title(symbol_data):
            current_symbol = symbol_data.get('symbol', "") if symbol_data else ""
            logger.info(f"Updating title with symbol: {current_symbol}")
            return f"Live Market Depth "

        # Callback to handle start button
        @self.app.callback(
            [Output('pause-button', 'style'),
             Output('app-state', 'data'),
             Output('interval-component', 'disabled'),
             Output('symbol-store', 'data'),
             Output('status-message', 'children'),
             Output('live-bookmap', 'figure')],
            [Input('start-button', 'n_clicks')],
            [State('symbol-input', 'value'),
             State('app-state', 'data'),
             State('zoom-store', 'data')]
        )
        def handle_start(n_clicks, symbol_value, app_state, current_zoom):
            if n_clicks is None:
                raise exceptions.PreventUpdate
            
            logger.info(f"Start button clicked. Symbol value: {symbol_value}")
            
            if not symbol_value:
                return {'display': 'none'}, \
                       {'is_paused': True, 'is_started': False}, True, {'symbol': ""}, \
                       "Please enter a symbol", no_update
            
            try:
                # Try to update the symbol (this will also subscribe to RTD)
                self.update_symbol(symbol_value)
                
                # Create a new figure with the updated symbol
                fig = bookmap_viz.create_bookmap(
                    db=self.db,
                    symbol=symbol_value,
                    start_time=time.time() - 30,
                    end_time=time.time()
                )
                
                # Show controls and start updates
                return {'display': 'block'}, \
                       {'is_paused': False, 'is_started': True}, False, \
                       {'symbol': symbol_value}, f"Started monitoring {symbol_value}", fig
                       
            except Exception as e:
                logger.error(f"Error starting symbol subscription: {e}")
                # Create an error figure
                error_fig = bookmap_viz.create_bookmap_figure(title="Error")
                error_fig.add_annotation(
                    text=f"Error: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=24, color="red")
                )
                return {'display': 'none'}, \
                       {'is_paused': True, 'is_started': False}, True, {'symbol': ""}, \
                       f"Error: {str(e)}", error_fig

        # Callback to handle pause/resume button
        @self.app.callback(
            [Output('pause-button', 'children'),
             Output('app-state', 'data', allow_duplicate=True),
             Output('interval-component', 'disabled', allow_duplicate=True)],
            [Input('pause-button', 'n_clicks')],
            [State('app-state', 'data')],
            prevent_initial_call=True
        )
        def toggle_pause(n_clicks, app_state):
            if n_clicks is None:
                raise exceptions.PreventUpdate
            
            try:
                current_state = app_state if isinstance(app_state, dict) else {'is_paused': True, 'is_started': False}
                is_paused = not current_state.get('is_paused', True)
                new_state = {'is_paused': is_paused, 'is_started': current_state.get('is_started', False)}
                logger.info(f"Toggle pause - New state: {new_state}")
                return 'Resume' if is_paused else 'Pause', new_state, is_paused
            except Exception as e:
                logger.error(f"Error in toggle_pause: {e}")
                return 'Pause', {'is_paused': True, 'is_started': False}, True

        # Update the existing figure callback to respect pause state and symbol updates
        @self.app.callback(
            [Output('live-bookmap', 'figure', allow_duplicate=True),
             Output('status-message', 'children', allow_duplicate=True)],
            [Input('interval-component', 'n_intervals'),
             Input('zoom-store', 'data'),
             Input('symbol-store', 'data')],
            [State('app-state', 'data'),
             State('live-bookmap', 'figure')],
            prevent_initial_call=True
        )
        def update_bookmap(n_intervals, zoom_data, symbol_data, app_state, current_fig):
            try:
                # If paused, don't update the figure
                if app_state and isinstance(app_state, dict) and app_state.get('is_paused', False):
                    raise exceptions.PreventUpdate
                
                # Make sure database file exists and is accessible
                if not self.db or not self.db.conn:
                    logger.warning("No database connection, trying to reconnect")
                    self.init_database()
                
                # Check database file exists
                if not os.path.exists(self.db.db_path):
                    logger.error(f"Database file not found: {self.db.db_path}")
                    return go.Figure(), "Database file not found. Check RTD connection."
                
                # Get current symbol from store
                current_symbol = symbol_data.get('symbol', self.symbol) if symbol_data else self.symbol
                
                # Check if we have any market depth data
                try:
                    cursor = self.db.conn.execute(
                        "SELECT COUNT(*) FROM market_depth WHERE symbol = ?",
                        (current_symbol,)
                    )
                    count = cursor.fetchone()[0]
                except Exception as e:
                    logger.error(f"Error querying market depth data: {e}")
                    return go.Figure(), f"Error querying database: {str(e)}"
                
                if count == 0:
                    logger.warning("No market depth data available yet")
                    # Create empty figure with proper styling
                    fig = bookmap_viz.create_bookmap_figure(title=f"Waiting for data - {current_symbol}")
                    
                    fig.add_annotation(
                        text="Waiting for market depth data...",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=24, color="white")
                    )
                    return fig, "Waiting for market depth data..."
                
                logger.info(f"Found {count} market depth records for {current_symbol}")
                
                # Check if current_fig is a proper Plotly figure object
                valid_fig = (current_fig is not None and 
                            isinstance(current_fig, go.Figure) and 
                            hasattr(current_fig, 'data'))
                
                # If we don't have a valid figure or symbol changed, create a new one
                if (not valid_fig or 
                    (valid_fig and len(current_fig.data) < 2) or 
                    (zoom_data is not None and 'symbol_change' in zoom_data)):
                    # Create the bookmap visualization from scratch
                    end_time = time.time() # UTC epoch seconds

                    # Load all data by not specifying start_time
                    # Set the display range to last 60 seconds
                    display_start = end_time - 60  # UTC epoch seconds
                    
                    # Get current y-axis range if available
                    current_range = None
                    if valid_fig and hasattr(current_fig, 'layout') and hasattr(current_fig.layout, 'yaxis'):
                        current_range = current_fig.layout.yaxis.range
                        logger.debug(f"Passing current range to new bookmap: {current_range}")
                    
                    fig = bookmap_viz.create_bookmap(
                        db=self.db,
                        symbol=current_symbol,
                        start_time=None,  # Load all data
                        end_time=end_time,
                        current_range=current_range
                    )
                    
                    # Set the x-axis range to show only the last 60 seconds in EST
                    est_tz = pytz.timezone('US/Eastern')
                    display_start_dt = pd.Timestamp(display_start, unit='s', tz='UTC').tz_convert(est_tz)
                    display_end_dt = pd.Timestamp(end_time, unit='s', tz='UTC').tz_convert(est_tz)
                    fig.update_xaxes(range=[display_start_dt, display_end_dt])
                    
                else:
                    # Update the existing figure
                    if not valid_fig:
                        logger.warning("Unexpected invalid figure in update_bookmap, creating new one")
                        fig = bookmap_viz.create_bookmap(
                            db=self.db,
                            symbol=current_symbol,
                            start_time=None,  # Load all data
                            end_time=time.time()
                        )
                    else:
                        fig = bookmap_viz.update_bookmap_live(
                            fig=current_fig,
                            db=self.db,
                            symbol=current_symbol,
                            lookback_seconds=60  # Match the default display window
                        )
                
                # Apply saved zoom state if available
                if zoom_data and isinstance(zoom_data, dict):
                    logger.debug(f"Applying saved zoom state: {zoom_data}")
                    try:
                        if 'xaxis.range[0]' in zoom_data and 'xaxis.range[1]' in zoom_data:
                            fig.update_layout(
                                xaxis=dict(
                                    range=[zoom_data['xaxis.range[0]'], zoom_data['xaxis.range[1]']]
                                )
                            )
                        if 'yaxis.range[0]' in zoom_data and 'yaxis.range[1]' in zoom_data:
                            fig.update_layout(
                                yaxis=dict(
                                    range=[zoom_data['yaxis.range[0]'], zoom_data['yaxis.range[1]']]
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Error applying zoom state: {e}")
                        # Continue without applying zoom state
                
                # Update status message (already uses EST)
                est_now = datetime.now(pytz.timezone('US/Eastern'))
                return fig, f"Last update: {est_now.strftime('%H:%M:%S %Z')} - {count} records"
                
                
            except Exception as e:
                logger.error(f"Error updating bookmap: {e}", exc_info=True)
                # Update status message with error details
                est_now = datetime.now(pytz.timezone('US/Eastern'))
                return go.Figure(), f"Error at {est_now.strftime('%H:%M:%S %Z')}: {str(e)}"
        
        # Clientside callback to store zoom level when user interacts with chart
        self.app.clientside_callback(
            """
            function(relayoutData, previousZoomData) {
                // Initialize the zoomProps with the previous zoom data
                const zoomProps = previousZoomData || {};
                
                // Only update when we have new data
                if(relayoutData === undefined || Object.keys(relayoutData).length === 0) {
                    return zoomProps;
                }
                
                // Check for standard zoom range
                if('xaxis.range[0]' in relayoutData) {
                    zoomProps['xaxis.range[0]'] = relayoutData['xaxis.range[0]'];
                    zoomProps['xaxis.range[1]'] = relayoutData['xaxis.range[1]'];
                }
                
                if('yaxis.range[0]' in relayoutData) {
                    zoomProps['yaxis.range[0]'] = relayoutData['yaxis.range[0]'];
                    zoomProps['yaxis.range[1]'] = relayoutData['yaxis.range[1]'];
                }
                
                // Check for autorange changes
                if('xaxis.autorange' in relayoutData) {
                    if(relayoutData['xaxis.autorange']) {
                        // If autorange is enabled, remove the manual ranges
                        delete zoomProps['xaxis.range[0]'];
                        delete zoomProps['xaxis.range[1]'];
                    }
                    zoomProps['xaxis.autorange'] = relayoutData['xaxis.autorange'];
                }
                
                if('yaxis.autorange' in relayoutData) {
                    if(relayoutData['yaxis.autorange']) {
                        // If autorange is enabled, remove the manual ranges
                        delete zoomProps['yaxis.range[0]'];
                        delete zoomProps['yaxis.range[1]'];
                    }
                    zoomProps['yaxis.autorange'] = relayoutData['yaxis.autorange'];
                }
                
                // Check for auto size (double click reset)
                if('autosize' in relayoutData) {
                    // Reset zoom on double click
                    delete zoomProps['xaxis.range[0]'];
                    delete zoomProps['xaxis.range[1]'];
                    delete zoomProps['yaxis.range[0]'];
                    delete zoomProps['yaxis.range[1]'];
                    zoomProps['autosize'] = true;
                }
                
                // Don't clear the symbol_change flag if it exists
                // to ensure it's preserved for the next callback
                return zoomProps;
            }
            """,
            Output('zoom-store', 'data', allow_duplicate=True),
            [Input('live-bookmap', 'relayoutData')],
            [State('zoom-store', 'data')],
            prevent_initial_call=True
        )

        # Add a separate callback to initialize symbol input with the last used symbol
        @self.app.callback(
            Output('symbol-input', 'value', allow_duplicate=True),
            Input('store-symbol-clientside', 'data'),
            prevent_initial_call=True
        )
        def init_symbol_from_storage(stored_data):
            # This just ensures the value from localStorage is respected
            # The actual loading of the value is done via the clientside JavaScript
            raise exceptions.PreventUpdate
    
    def run(self, debug=False, port=8050, host='127.0.0.1'):
        """Run the Dash application."""
        try:
            logger.info(f"Starting Dash server on {host}:{port}")
            # Open browser after a short delay
            Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
            # Fix: Use app.run() instead of app.run_server()
            self.app.run(debug=debug, port=port, host=host)
        except Exception as e:
            logger.error(f"Error running Dash server: {e}")
            raise

def create_dash_app(symbol: str, update_interval: int = 2) -> BookmapDashApp:
    """Create and return a BookmapDashApp instance."""
    return BookmapDashApp(symbol, update_interval)
