# TOS Market Depth RTD

A Bookmap-style visualization tool for displaying real-time market depth data from ThinkorSwim's RTD server. 

This is a rough v1. Improvements coming.

![](https://github.com/user-attachments/assets/ed78e5ea-8d38-48fa-883a-f53a3b072b1e)

https://github.com/user-attachments/assets/0f836aac-8b71-425e-9b22-9530e8dcd742


## Features

- Real-time market depth visualization in Bookmap style
- Stores RTD data in SQLite database for persistence and analysis

## Installation

### Prerequisites

- Python 3.11+
- ThinkorSwim desktop application
- Windows OS (COM components are Windows-specific)

### Setup

1. Clone the repository
```
git clone https://github.com/2187Nick/tos-market-depth-rtd.git
cd tos-market-depth-rtd
```

2. Create a virtual environment
```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

## Usage

1. Make sure ThinkorSwim is running and you're logged in

2. Start the application
```
python app.py
```

3. Enter an option symbol in the interface:
```
Examples:
Stock options: .SPY250417C550
E-mini S&P 500 Index Future Options: ./EWJ25C5725:XCME
Nasdaq 100 Index Options: .NDX250417C20000
Oil Future Options: ./LO4H25C70:XNYM
10-Year Us Treasury Note Futures Options: ./ZN2J25C110.75:XCBT
```

``` 
To get the symbol:
Go to the option chain page.
Right click on any option and then click "copy" on symbol name.
```

4. Click "Start" to begin visualization

5. Use the controls to:
   - Pause/Resume the visualization
   - Update the symbol
   - Zoom in/out of specific areas

## Architecture

### Core Components

- **RTD Client** (`src/rtd/client.py`): Interfaces with the ThinkorSwim RTD server via COM to receive market data
- **Database** (`src/db/database.py`): SQLite implementation for storing market depth data
- **Bookmap Visualization** (`src/utils/bookmap_viz.py`): Creates and updates the heatmap visualization
- **Dash App** (`dash_app.py`): Web interface built with Dash and Plotly

### Data Flow

1. The RTD Client connects to ThinkorSwim and subscribes to market depth data for specified symbols
2. Data is captured and stored in the SQLite database
3. The visualization module pulls data from the database and creates the heatmap
4. The Dash web app provides the user interface and refreshes the visualization at regular intervals

## Configuration

You can adjust settings in the `config/settings.json` file:

- RTD server settings
- Database configurations
- Visualization parameters
- Update intervals

## Credit

[@2187Nick](https://x.com/2187Nick)

[Discord](https://discord.com/invite/vxKepZ6XNC)

Backend:

[PYRTDC](https://github.com/tifoji/pyrtdc/)

## Notes

1. This does not capture all trades or all market liquidity.
2. The API version captures more market liquidity(A combo version might be coming..).
3. This is experimental and mainly just for exploring what's possible.


