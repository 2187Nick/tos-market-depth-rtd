# src/utils/bookmap_viz.py
"""
Bookmap-style visualization utilities for market depth data.

This module provides functions to visualize market depth data in a 
bookmap-style heatmap format, with time on the x-axis and price levels
on the y-axis, showing bid and ask sizes through color intensity.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import pytz

from src.db.database import BookmapDatabase
from src.core.logger import get_logger

logger = get_logger(__name__)

# Color constants with brighter colors for visualization
BID_COLORSCALE = [[0, 'rgba(0,255,0,0)'], [0.1, 'rgba(0,255,0,0.3)'], [0.5, 'rgba(0,255,0,0.7)'], [1, 'rgba(0,255,0,1)']]  # Green gradient
ASK_COLORSCALE = [[0, 'rgba(255,0,0,0)'], [0.1, 'rgba(255,0,0,0.3)'], [0.5, 'rgba(255,0,0,0.7)'], [1, 'rgba(255,0,0,1)']]  # Red gradient

# Constants for trade size circle scaling
MIN_CIRCLE_SIZE = 20  # Minimum circle size
MAX_CIRCLE_SIZE = 80  # Maximum circle size
SIZE_SCALE_FACTOR = 0.5  # Adjusts overall scaling


def create_bookmap_figure(title: str = "Market Depth Bookmap") -> go.Figure:
    """
    Create an empty bookmap figure with appropriate layout.
    
    Args:
        title: Title for the bookmap figure
        
    Returns:
        Plotly figure object ready for bookmap data
    """
    fig = go.Figure()
    
    # Add an invisible trace to ensure the plot renders properly
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (EST)",  # Updated to show EST timezone
        yaxis_title="",  # Remove y-axis title since we'll have a price column
        height=900,  # Default height increased
        width=1600,  # Default width increased
        plot_bgcolor='rgb(0,0,0)',  # Pure black background
        paper_bgcolor='rgb(0,0,0)',  # Pure black background
        font=dict(color='white', size=20),  # Reduced base font size
        margin=dict(l=50, r=200, t=50, b=50),  # Increased right margin for price and size columns
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=20)  # Reduced legend font size
        ),
        # Hide y-axis labels since we'll have a price column
        yaxis=dict(
            autorange=False,
            fixedrange=False,
            showgrid=True,
            gridcolor='rgba(80,80,80,0.3)',
            gridwidth=1,
            side='right',  # Move price axis to right side
            showticklabels=False,  # Hide the y-axis tick labels
            ticksuffix="  ",  # Reduced space as we'll use separate annotations
            constrain='domain',  # This constrains the axis to the domain
            constraintoward='center',  # Centers the constraint
            scaleanchor='y',  # This ensures consistent scaling
            tickfont=dict(size=20),  # Added explicit tick font size
            range=[0, 1]  # Set default range to ensure proper rendering
        ),
        # Time axis
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(80,80,80,0.3)',
            gridwidth=1,
            tickfont=dict(size=20),  # Added explicit tick font size
            range=[datetime.now(pytz.timezone('US/Eastern')) - timedelta(minutes=5), 
                  datetime.now(pytz.timezone('US/Eastern'))]  # Set default range
        )
    )
    
    return fig


def fetch_market_depth_data(db: BookmapDatabase, 
                           symbol: str,
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None) -> pd.DataFrame:
    """
    Fetch market depth data and convert to DataFrame format suitable for visualization.
    Also identifies LAST price points which have bid_size=0 and ask_size=0.
    
    Args:
        db: Database instance to query
        symbol: Symbol to get depth data for
        start_time: Optional start timestamp. If None, will fetch ALL available data
        end_time: Optional end timestamp
        
    Returns:
        DataFrame with market depth data
    """
    # Set default time range if not provided
    if end_time is None:
        end_time = time.time()
        print(f"end_time: {end_time}")
    
    # Note: We no longer set a default start_time, allowing all data to be fetched if not specified
    
    # Build the SQL query based on whether start_time is provided
    if start_time is not None:
        query = """
            SELECT symbol, price, bid_size, ask_size, timestamp,
                   CASE WHEN bid_size = 0 AND ask_size = 0 THEN 1 ELSE 0 END as is_last
            FROM market_depth 
            WHERE symbol = ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC, price DESC
        """
        params = (symbol, start_time, end_time)
    else:
        query = """
            SELECT symbol, price, bid_size, ask_size, timestamp,
                   CASE WHEN bid_size = 0 AND ask_size = 0 THEN 1 ELSE 0 END as is_last
            FROM market_depth 
            WHERE symbol = ?
            AND timestamp <= ?
            ORDER BY timestamp ASC, price DESC
        """
        params = (symbol, end_time)
    
    cursor = db.conn.execute(query, params)
    depth_data = cursor.fetchall()
    
    if not depth_data:
        logger.warning(f"No market depth data found for {symbol}")
        return pd.DataFrame(columns=['symbol', 'timestamp', 'price', 'bid_size', 'ask_size', 'is_last'])
    
    # Convert to DataFrame
    df = pd.DataFrame(depth_data, columns=['symbol', 'price', 'bid_size', 'ask_size', 'timestamp', 'is_last'])
    
    # Convert timestamp to datetime for better plotting
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    # Convert to US/Eastern timezone
    est_tz = pytz.timezone('US/Eastern')
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(est_tz)
    
    logger.info(f"Retrieved {len(df)} market depth records for {symbol}")
    
    return df


def fetch_last_size_data(db: BookmapDatabase, 
                        symbol: str,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None) -> pd.DataFrame:
    """
    Fetch LAST_SIZE trade data from the database.
    
    Args:
        db: Database instance to query
        symbol: Symbol to get trade data for
        start_time: Optional start timestamp
        end_time: Optional end timestamp
        
    Returns:
        DataFrame with trade size data
    """
    # Set default time range if not provided
    if end_time is None:
        end_time = time.time()
    if start_time is None:
        start_time = end_time - 1000 #3600  # 1 hour of data by default
    
    # Get trade size data from database
    cursor = db.conn.execute("""
        SELECT price, size, timestamp
        FROM last_size_history 
        WHERE symbol = ?
        AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """, (symbol, start_time, end_time))
    
    size_data = cursor.fetchall()
    
    if not size_data:
        logger.warning(f"No trade size data found for {symbol}")
        return pd.DataFrame(columns=['price', 'size', 'timestamp'])
    
    # Convert to DataFrame
    df = pd.DataFrame(size_data, columns=['price', 'size', 'timestamp'])
    
    # Convert timestamp to datetime for better plotting
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    # Convert to US/Eastern timezone
    est_tz = pytz.timezone('US/Eastern')
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(est_tz)
    
    logger.info(f"Retrieved {len(df)} trade size records for {symbol}")
    
    return df


def safe_min_max_prices(x, df):
    """Helper function to safely get min ask and max bid prices."""
    ask_mask = df.loc[x.index, 'ask_size'] > 0
    bid_mask = df.loc[x.index, 'bid_size'] > 0
    
    ask_prices = x[ask_mask]
    bid_prices = x[bid_mask]
    
    min_ask = None
    max_bid = None
    
    try:
        if not ask_prices.empty:
            min_ask = min(ask_prices)
    except (ValueError, TypeError):
        pass
        
    try:
        if not bid_prices.empty:
            max_bid = max(bid_prices)
    except (ValueError, TypeError):
        pass
        
    return (min_ask, max_bid)

def prepare_heatmap_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data for the bid and ask heatmaps.
    
    Args:
        df: DataFrame with market depth data
        
    Returns:
        Tuple containing (times, prices, bid_sizes, ask_sizes)
    """
    if df.empty:
        return [], [], np.array([[]]), np.array([[]])
    
    # Get unique timestamps and prices
    times = df['datetime'].unique()
    prices = sorted(df['price'].unique(), reverse=True)  # Sort prices in descending order
    
    # Ensure we have consistent dimensions
    if len(times) == 0 or len(prices) == 0:
        logger.warning("No valid times or prices found in data")
        return times, prices, np.array([[]]), np.array([[]])
    
    # Create matrices for bid and ask sizes
    bid_sizes = np.zeros((len(prices), len(times)))
    ask_sizes = np.zeros((len(prices), len(times)))
    
    # Add a small epsilon to ensure no zero values (helps with colorscale visualization)
    epsilon = 1e-10
    
    # Map of price to row index and timestamp to column index
    price_to_idx = {price: i for i, price in enumerate(prices)}
    # Use the timezone-aware datetime objects directly as keys
    time_to_idx = {time: i for i, time in enumerate(times)}
    
    # Debug counters
    bid_count = 0
    ask_count = 0
    
    # Fill matrices with size data
    for _, row in df.iterrows():
        price_idx = price_to_idx.get(row['price'])
        time_idx = time_to_idx.get(row['datetime'])
        
        if price_idx is not None and time_idx is not None:
            if 'bid_size' in row and row['bid_size'] > 0:
                bid_sizes[price_idx, time_idx] = row['bid_size']
                bid_count += 1
            if 'ask_size' in row and row['ask_size'] > 0:
                ask_sizes[price_idx, time_idx] = row['ask_size']
                ask_count += 1
    
    # Log the counts for debugging
    logger.info(f"Prepared heatmap data with {len(prices)} price levels, {len(times)} time points")
    logger.info(f"Found {bid_count} bid size values and {ask_count} ask size values")
    
    # Ensure non-zero minimum values for better visualization
    if bid_count > 0 and np.max(bid_sizes) > 0:
        # Replace zeros with small epsilon to help visualization
        bid_sizes[bid_sizes == 0] = epsilon
    
    if ask_count > 0 and np.max(ask_sizes) > 0:
        # Replace zeros with small epsilon to help visualization
        ask_sizes[ask_sizes == 0] = epsilon
    
    return times, prices, bid_sizes, ask_sizes


def calculate_y_axis_range(prices: List[float], df: pd.DataFrame, db: BookmapDatabase, lookback_seconds: int = 30, current_range: Optional[List[float]] = None) -> tuple:
    """
    Calculate y-axis range based on the latest BID and ASK quotes from latest_quotes table.
    Uses padding to create a visible range around the current spread.
    
    If current_range is provided, it will only make small adjustments to prevent jarring changes.
    
    Args:
        prices: List of price levels (used as fallback)
        df: DataFrame with market depth data (used as fallback)
        db: Database connection to query latest quotes
        lookback_seconds: Number of seconds to look back for range calculation
        current_range: Current y-axis range [min, max] if available
        
    Returns:
        Tuple of (y_min, y_max)
    """
    if not prices:
        # If no prices available, use a small sensible default range instead of [0, 1]
        return 0.25, 0.35
    
    
    # Get the latest BID and ASK from latest_quotes table
    symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else None
    if not symbol:
        #print(f"No symbol found in df")
        # If we can't get the symbol, fall back to using price range
        max_price = max(prices)
        min_price = min(prices)
        price_range = max_price - min_price
        return min_price - price_range * 0.20, max_price + price_range * 0.20
    
    # Query latest quotes for BID and ASK
    cursor = db.conn.execute("""
        SELECT quote_type, value 
        FROM latest_quotes 
        WHERE symbol = ? AND quote_type IN ('BID', 'ASK')
        ORDER BY timestamp DESC
    """, (symbol,))
    
    quotes = cursor.fetchall()
    latest_quotes = dict(quotes)
    
    bid_price = latest_quotes.get('BID')
    ask_price = latest_quotes.get('ASK')
    
    if bid_price is None or ask_price is None:
        # If we can't get both BID and ASK, fall back to using price range
        max_price = max(prices)
        min_price = min(prices)
        price_range = max_price - min_price
        return min_price - price_range * 0.20, max_price + price_range * 0.20
    

    
    # Calculate padding based on bid price 
    # Set to 20% of bid price or 2 ticks, whichever is greater
    padding = max(bid_price * .2, 0.02)
    #print(f"padding: {padding}")
    
    y_min = bid_price - padding
    y_max = ask_price + padding
    #print(f"Bid / ASk y_min: {y_min}, y_max: {y_max}")
    
    # Ensure minimum range size for visibility - especially important for options
    range_size = y_max - y_min
    min_range_size = 0.03 if min(prices) < 1.0 else 0.1  # Different minimums based on price scale
    
    if range_size < min_range_size:
        # Center the range around the middle point and expand
        mid_point = (y_min + y_max) / 2
        y_min = mid_point - min_range_size / 2
        y_max = mid_point + min_range_size / 2
    
    # If we have a current range, ensure smooth transitions
    if current_range and len(current_range) == 2:
        current_min, current_max = current_range
        
        # If current range is the default [0,1], don't use it for transitions
        if not (current_min == 0 and current_max == 1):
            # Don't make extreme jumps in the range - limit to 20% of the current view
            curr_range_size = current_max - current_min
            max_change = curr_range_size * 0.20
            
            # Smooth transition by only adjusting if necessary
            if y_min > current_min and y_min < current_max:
                # New min is in the current view, keep the current min
                y_min = current_min
            elif abs(y_min - current_min) > max_change:
                # Limit the change if it's too drastic
                y_min = current_min + (1 if y_min > current_min else -1) * max_change
                
            if y_max < current_max and y_max > current_min:
                # New max is in the current view, keep the current max
                y_max = current_max
            elif abs(y_max - current_max) > max_change:
                # Limit the change if it's too drastic
                y_max = current_max + (1 if y_max > current_max else -1) * max_change
    
    #print(f"Final y_min: {y_min}, y_max: {y_max}")
    return y_min, y_max


def create_bookmap(db: BookmapDatabase, 
                 symbol: str,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None,
                 size_scaling: float = 0.05,
                 current_range: Optional[List[float]] = None) -> go.Figure:
    """
    Create a bookmap visualization for market depth data.
    
    Args:
        db: Database instance
        symbol: Symbol to display
        start_time: Start timestamp (default: end_time - 1 hour)
        end_time: End timestamp (default: current time)
        size_scaling: Size scaling factor
        current_range: Optional current y-axis range to preserve
        
    Returns:
        Plotly figure with bookmap visualization
    """
    logger.info(f"Creating bookmap for symbol: {symbol}, start_time: {start_time}, end_time: {end_time}")
    
    # Create figure with title including current price
    fig = create_bookmap_figure(title=f"{symbol}")
    
    # Get market depth data
    df = fetch_market_depth_data(db, symbol, start_time, end_time)
    
    # Get trade size data
    trade_df = fetch_last_size_data(db, symbol, start_time, end_time)
    
    if df.empty:
        logger.warning(f"No market depth data available for symbol: {symbol}")
        fig.add_annotation(
            text="No market depth data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="white")
        )
        return fig
    
    logger.info(f"Retrieved {len(df)} market depth records for {symbol}")
    
    # Prepare data for heatmaps
    times, prices, bid_sizes, ask_sizes = prepare_heatmap_data(df)
    
    if len(times) == 0 or len(prices) == 0:
        logger.warning(f"Insufficient data to generate bookmap for {symbol}")
        fig.add_annotation(
            text="Insufficient data to generate bookmap",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="white")
        )
        return fig
    
    logger.info(f"Prepared heatmap data with {len(times)} time points and {len(prices)} price levels")
    
    # Scale sizes for better visualization
    max_bid_size = np.max(bid_sizes) if np.max(bid_sizes) > 0 else 1
    max_ask_size = np.max(ask_sizes) if np.max(ask_sizes) > 0 else 1
    
    logger.info(f"Max bid size: {max_bid_size}, Max ask size: {max_ask_size}")
    
    # Ensure there are actual values to display
    if max_bid_size <= 0 and max_ask_size <= 0:
        logger.warning(f"No market depth sizes available for {symbol}")
        fig.add_annotation(
            text="No market depth sizes available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="white")
        )
        return fig
    
    # Make sure we have some non-zero values by adding small epsilon to empty cells
    epsilon = 1e-10

    #print(f"bid_sizes: {bid_sizes}")
    #print(f"ask_sizes: {ask_sizes}")

    bid_sizes = np.maximum(bid_sizes, epsilon)
    ask_sizes = np.maximum(ask_sizes, epsilon)
    
    
    # Use the larger of the two maximums for consistent scaling
    max_size = max(max_bid_size, max_ask_size)
    logger.info(f"Updating with max bid size: {max_bid_size}, max ask size: {max_ask_size}, using max: {max_size}")
    
    # Normalize sizes for color intensity
    normalized_bid_sizes = bid_sizes / max_size
    normalized_ask_sizes = ask_sizes / max_size
    
    # Add bid size heatmap
    fig.add_trace(go.Heatmap(
        x=times,
        y=prices,
        z=normalized_bid_sizes,
        colorscale=BID_COLORSCALE,
        showscale=False,
        name=f"Bid Size (Max: {int(max_bid_size)})",
        customdata=bid_sizes.astype(int),
        hovertemplate='Time: %{x}<br>Price: %{y}<br><extra></extra>',
        zmin=0,
        zmax=1,
        yhoverformat='.2f'
    ))
    
    # Add ask size heatmap
    fig.add_trace(go.Heatmap(
        x=times,
        y=prices,
        z=normalized_ask_sizes,
        colorscale=ASK_COLORSCALE,
        showscale=False,
        name=f"Ask Size (Max: {int(max_ask_size)})",
        customdata=ask_sizes.astype(int),
        hovertemplate='Time: %{x}<br>Price: %{y}<br><extra></extra>',
        zmin=0,
        zmax=1,
        yhoverformat='.2f'
    ))

    # Calculate y-axis range based on both price and size criteria
    y_min, y_max = calculate_y_axis_range(prices, df, db, lookback_seconds=30, current_range=current_range)

    # Update layout with fixed sizes
    fig.update_layout(
        yaxis=dict(
            scaleanchor='y',  # This ensures consistent scaling
            constrain='domain',  # This constrains the axis to the domain
            constraintoward='center',  # Centers the constraint
            range=[y_min, y_max],
            showticklabels=False,  # Hide the y-axis tick labels
            gridwidth=1,
            gridcolor='rgba(80,80,80,0.3)'
        )
    )

    # Use fixed font sizes instead of calculating them
    price_font_size = 20  # Reduced from 10
    size_font_size = 20   # Reduced from 10
    trade_font_size = 20   # Reduced from 10

    # Add trade size markers with dynamic size based on trade size
    if not trade_df.empty:
        # Calculate scaled sizes for circles based on trade size
        max_trade_size = trade_df['size'].max()
        min_trade_size = trade_df['size'].min()
        
        # Prevent division by zero and ensure minimum variation
        size_range = max(max_trade_size - min_trade_size, 1)
        
        # Scale sizes between MIN_CIRCLE_SIZE and MAX_CIRCLE_SIZE
        scaled_sizes = MIN_CIRCLE_SIZE + (trade_df['size'] - min_trade_size) / size_range * (MAX_CIRCLE_SIZE - MIN_CIRCLE_SIZE) * SIZE_SCALE_FACTOR
        
        # Ensure minimum size for visibility
        scaled_sizes = np.maximum(scaled_sizes, MIN_CIRCLE_SIZE)
        
        fig.add_trace(go.Scatter(
            x=trade_df['datetime'],
            y=trade_df['price'],
            mode='markers+text',
            marker=dict(
                color='yellow',  # Changed from yellow to cyan
                size=scaled_sizes,  # Using dynamically scaled sizes
                opacity=0.7,  # Increased opacity slightly
                line=dict(color='rgb(50,50,50)', width=1)
            ),
            text=trade_df['size'].astype(int).astype(str),
            textposition='middle center',
            textfont=dict(
                color='black',
                size=trade_font_size,
                family='Arial Black'
            ),
            name='Trade Size',
            hovertemplate='Trade Size: %{text}<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>',
            showlegend=True
        ))

    # Get current best bid and ask for title
    if prices:
        max_price = max(prices)
        min_price = min(prices)
        price_range = max_price - min_price
        y_min = min_price - price_range * 0.10
        y_max = max_price + price_range * 0.10
    else:
        # Try to get current range from figure
        try:
            current_range = fig.layout.yaxis.range
            if current_range and len(current_range) == 2:
                y_min, y_max = current_range
            else:
                # If no current range, try to get from recent data
                recent_df = df[df['timestamp'] >= df['timestamp'].max() - 30]  # Last 30 seconds
                if not recent_df.empty:
                    max_price = recent_df['price'].max()
                    min_price = recent_df['price'].min()
                    price_range = max_price - min_price
                    y_min = min_price - price_range * 0.10
                    y_max = max_price + price_range * 0.10
                else:
                    # Last resort fallback
                    y_min = 0
                    y_max = 1
        except Exception as e:
            logger.warning(f"Error getting y-axis range: {e}")
            y_min = 0
            y_max = 1

    # Add price column and size values
    annotations = []
    
    for price in prices:
        price_idx = prices.index(price)
        
        # Get latest bid and ask sizes for this price level
        if len(bid_sizes) > 0 and len(bid_sizes[price_idx]) > 0:
            latest_bid = bid_sizes[price_idx][-1]
        else:
            latest_bid = 0
            
        if len(ask_sizes) > 0 and len(ask_sizes[price_idx]) > 0:
            latest_ask = ask_sizes[price_idx][-1] 
        else:
            latest_ask = 0
        
        # Price column (column 1)
        annotations.append(dict(
            x=1.05,  # Position for price labels
            y=price,
            xref='paper',
            yref='y',
            text=f"{price:.2f}",
            showarrow=False,
            font=dict(size=price_font_size, color='white'),
            align='right'
        ))
        
        # Size column (column 2) - Only add annotation if size is significantly non-zero
        if latest_ask > epsilon:  # Only show if truly non-zero (not epsilon)
            # Show only ask size (red text)
            annotations.append(dict(
                x=1.15,  # Position for size labels
                y=price,
                xref='paper',
                yref='y',
                text=f"{int(latest_ask)}",
                showarrow=False,
                font=dict(size=size_font_size, color='red'),
                align='center'
            ))
        elif latest_bid > epsilon:  # Only show if truly non-zero (not epsilon)
            # Show only bid size (green text)
            annotations.append(dict(
                x=1.15,  # Position for size labels
                y=price,
                xref='paper',
                yref='y',
                text=f"{int(latest_bid)}",
                showarrow=False,
                font=dict(size=size_font_size, color='lime'),
                align='center'
            ))

    # Update layout with more visible borders and column headers
    fig.update_layout(
        annotations=annotations,
        shapes=[
            
            # Border between chart and price column
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=1.0,
                y0=0,
                x1=1.0,
                y1=1,
                line=dict(color="white", width=2)
            ),
            # Border between price and size columns
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=1.10,  # Adjusted from 1.12
                y0=0,
                x1=1.10,
                y1=1,
                line=dict(color="white", width=2)
            ),
            # Border after size column
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=1.20,  # Reduced from 1.25 to 1.20
                y0=0,
                x1=1.20,
                y1=1,
                line=dict(color="white", width=2)
            )
        ]
    )
    
    # Add color scales as separate traces for legend - make sure to add after updating the best bid/ask lines
    fig.add_trace(go.Scatter(
        x=[times[0]],
        y=[prices[0]],
        mode='markers',
        marker=dict(color='green', size=30),
        name='Bid Size',
        showlegend=True,
        visible='legendonly'  # Hidden but shows in legend
    ))
    
    fig.add_trace(go.Scatter(
        x=[times[0]],
        y=[prices[-1]],
        mode='markers',
        marker=dict(color='red', size=30),
        name='Ask Size',
        showlegend=True,
        visible='legendonly'  # Hidden but shows in legend
    ))
    
    return fig


def update_bookmap_live(fig: go.Figure,
                      db: BookmapDatabase,
                      symbol: str,
                      lookback_seconds: int = 30,
                      force_y_update: bool = False) -> go.Figure:
    """
    Update a bookmap figure with the latest market depth data.
    
    Args:
        fig: The figure to update
        db: Database instance
        symbol: Symbol to display
        lookback_seconds: Number of seconds to look back for data
        force_y_update: Whether to force update the y-axis range
        
    Returns:
        Updated figure
    """
    end_time = time.time()
    start_time = end_time - lookback_seconds
    
    # Get updated market depth data
    df = fetch_market_depth_data(db, symbol, start_time, end_time)
    
    # Get updated trade size data
    trade_df = fetch_last_size_data(db, symbol, start_time, end_time)
    
    if df.empty:
        logger.warning("No market depth data available for update")
        return fig
    
    # Prepare new heatmap data
    times, prices, bid_sizes, ask_sizes = prepare_heatmap_data(df)
    
    if len(times) == 0 or len(prices) == 0:
        logger.warning("No time or price data available for update")
        return fig
    
    # Scale sizes for better visualization
    max_bid_size = np.max(bid_sizes) if np.max(bid_sizes) > 0 else 1
    max_ask_size = np.max(ask_sizes) if np.max(ask_sizes) > 0 else 1
    
    # Ensure there are actual values to display
    if max_bid_size <= 0 and max_ask_size <= 0:
        fig.add_annotation(
            text="No market depth sizes available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="white")
        )
        return fig
    
    # Make sure we have some non-zero values by adding small epsilon to empty cells
    epsilon = 1e-10

    #print(f"bid_sizes2: {bid_sizes}")
    #print(f"ask_sizes2: {ask_sizes}")
    bid_sizes = np.maximum(bid_sizes, epsilon)
    ask_sizes = np.maximum(ask_sizes, epsilon)
    
    # Use the larger of the two maximums for consistent scaling
    max_size = max(max_bid_size, max_ask_size)
    logger.info(f"Updating with max bid size: {max_bid_size}, max ask size: {max_ask_size}, using max: {max_size}")
    
    # Normalize sizes for color intensity
    normalized_bid_sizes = bid_sizes / max_size
    normalized_ask_sizes = ask_sizes / max_size
    
    # Update bid heatmap
    fig.data[0].update(
        x=times,
        y=prices,
        z=normalized_bid_sizes,
        customdata=bid_sizes.astype(int),
        colorscale=BID_COLORSCALE,
        showscale=False,
        zmin=0,
        zmax=1,
        yhoverformat='.2f',
        hovertemplate='Time: %{x}<br>Price: %{y}<br><extra></extra>'
    )
    
    # Update ask heatmap
    fig.data[1].update(
        x=times,
        y=prices,
        z=normalized_ask_sizes,
        customdata=ask_sizes.astype(int),
        colorscale=ASK_COLORSCALE,
        showscale=False,
        zmin=0,
        zmax=1,
        yhoverformat='.2f',
        hovertemplate='Time: %{x}<br>Price: %{y}<br><extra></extra>'
    )

    # Get current y-axis range
    current_range = fig.layout.yaxis.range
    
    # Only update the range if:
    # 1. We don't have a current range (first update)
    # 2. The current range is the default [0,1]
    # 3. The current range is invalid
    # 4. We're explicitly told to update it
    if (force_y_update or
        not current_range or 
        len(current_range) != 2 or 
        (current_range[0] == 0 and current_range[1] == 1) or
        not all(isinstance(x, (int, float)) for x in current_range)):
        
        # Calculate y-axis range based on recent data
        # Use a shorter lookback for y-axis range to make it more responsive
        y_min, y_max = calculate_y_axis_range(prices, df, db, lookback_seconds=30, current_range=current_range)
        #print(f"Update_bookmap_live y_min: {y_min}, y_max: {y_max}")
        
        # Use the calculated range
        new_y_min = y_min
        new_y_max = y_max
        
        # Update layout with new y-axis range
        fig.update_layout(
            yaxis=dict(
                range=[new_y_min, new_y_max],
                showticklabels=False,  # Hide the y-axis tick labels
                gridwidth=1,
                gridcolor='rgba(80,80,80,0.3)'
            )
        )
    # Otherwise keep the current range to maintain user's zoom level

    # Update trade size markers if they exist
    if not trade_df.empty and len(fig.data) > 2:  # Check if trade markers trace exists
        # Calculate scaled sizes for circles based on trade size
        max_trade_size = trade_df['size'].max()
        min_trade_size = trade_df['size'].min()
        
        # Prevent division by zero and ensure minimum variation
        size_range = max(max_trade_size - min_trade_size, 1)
        
        # Scale sizes between MIN_CIRCLE_SIZE and MAX_CIRCLE_SIZE
        scaled_sizes = MIN_CIRCLE_SIZE + (trade_df['size'] - min_trade_size) / size_range * (MAX_CIRCLE_SIZE - MIN_CIRCLE_SIZE) * SIZE_SCALE_FACTOR
        
        # Ensure minimum size for visibility
        scaled_sizes = np.maximum(scaled_sizes, MIN_CIRCLE_SIZE)
        
        fig.data[2].update(
            x=trade_df['datetime'],
            y=trade_df['price'],
            text=trade_df['size'].astype(int).astype(str),
            marker=dict(
                color='cyan',  # Changed from yellow to cyan
                size=scaled_sizes,  # Using dynamically scaled sizes
                opacity=0.7
            )
        )
    elif not trade_df.empty:  # Add trade markers if they don't exist
        # Calculate scaled sizes for circles based on trade size
        max_trade_size = trade_df['size'].max()
        min_trade_size = trade_df['size'].min()
        
        # Prevent division by zero and ensure minimum variation
        size_range = max(max_trade_size - min_trade_size, 1)
        
        # Scale sizes between MIN_CIRCLE_SIZE and MAX_CIRCLE_SIZE
        scaled_sizes = MIN_CIRCLE_SIZE + (trade_df['size'] - min_trade_size) / size_range * (MAX_CIRCLE_SIZE - MIN_CIRCLE_SIZE) * SIZE_SCALE_FACTOR
        
        # Ensure minimum size for visibility
        scaled_sizes = np.maximum(scaled_sizes, MIN_CIRCLE_SIZE)
        
        fig.add_trace(go.Scatter(
            x=trade_df['datetime'],
            y=trade_df['price'],
            mode='markers+text',
            marker=dict(
                color='white',  # Changed from yellow to cyan
                size=scaled_sizes,  # Using dynamically scaled sizes
                opacity=0.7,  # Increased opacity slightly
                line=dict(color='rgb(50,50,50)', width=1)
            ),
            text=trade_df['size'].astype(int).astype(str),
            textposition='middle center',
            textfont=dict(
                color='black',
                size=20,
                family='Arial Black'
            ),
            name='Trade Size',
            hovertemplate='Trade Size: %{text}<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>',
            showlegend=True
        ))

    return fig