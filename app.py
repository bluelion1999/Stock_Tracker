from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import yfinance as yf
import json
import os

import pandas
import numpy
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import joblib


# Create Flask app instance
app = Flask(__name__)

# Store stocks in memory    

app.secret_key = 'your-secret-key-for-flash-messages'

STOCKS_FILE = 'stocks.json'


def get_stock_data(symbol):
    """ Get real data from YF"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        #Get Curr Price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        #Get prev Close
        previous_close = info.get('previousClose')
        
        if current_price and previous_close:
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100

            return {
                'symbol' : symbol,
                'price' : f"{current_price:.2f}",
                'change' : f"{change:+.2f} ({change_percent:+.1f}%)",
                'valid' : True   
            }
        else:
            return {'valid': False, 'error': 'Could not fetch price data'}
    except Exception as e:
        return {'valid': False, 'error': f'Invalid symbol or API error: {str(e)}'}
    
    
def load_stocks():
    """Load stock symbols from file and fetch fresh data"""
    symbols = []
    if os.path.exists(STOCKS_FILE):
        try:
            with open(STOCKS_FILE, 'r') as f:
                symbols = json.load(f)
        except:
            return []
    
    # Fetch fresh data for each symbol
    stocks_data = []
    for symbol in symbols:
        stock_data = get_stock_data(symbol)
        if stock_data['valid']:
            stocks_data.append(stock_data)
        else:
            # Keep symbol even if API fails temporarily
            stocks_data.append({
                'symbol': symbol,
                'price': 'Error',
                'change': 'Error',
                'valid': False
            })
    
    return stocks_data

def save_stocks(stocks_data):
    """Save only stock symbols to file"""
    try:
        # Extract just the symbols
        symbols = [stock['symbol'] for stock in stocks_data]
        with open(STOCKS_FILE, 'w') as f:
            json.dump(symbols, f, indent=2)
    except Exception as e:
        print(f"Error saving stocks: {e}")
        
# Begin ML

class SimpleMLPredictor:
    """ML predictor for Stock Recommendations"""
    
    def __init__(self):
        self.is_trained = False
        print("ML Model is initialized (No training yet)")

    def predict_stock(self, symbol):
        """Placeholder for the moment for prediction"""
        if not self.is_trained:
            return None
        return {"recommendation": "HOLD", 'confidence' : 50.0}
    def calculate_technical_indicators(self,data):
        """Calculate key technical indicators"""
        print(f'Calculating indicators for {len(data)} datapoints')
        
        #Calculate 2 standard use SMAs
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()

        #Price to SMAs
        data['Price_to_SMA20'] = data['Close'] / data['SMA_20']
        data['Price_to_SMA50'] = data['Close'] / data['SMA_50']
        
        #RSI (momentum) 
        delta = data['Close'].diff() #x_n - x_n-1 or daily change
        gain = (delta.where(delta> 0,0)).rolling(window=14).mean() #time/amount in black
        loss = (-delta.where(delta< 0,0)).rolling(window=14).mean() #time/amount in red
        rs = gain/loss 
        data['RSI'] = 100 - (100 / (1 + rs))
        
        #Adding Vol measure (Bollinger bands) lmao this is just 95% conf band why not just call it what it is
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2) 
        #Transform applied to get relative position in the band if 0 @ lower band if 1 @ upper band 
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        #Momentum (% change over certain periods of time)
        data['Momentum_1M'] = data['Close'].pct_change(20)
        data['Momentum_3M'] = data['Close'].pct_change(60)
        data['Momentum_6M'] = data['Close'].pct_change(120)
        
        #Vol
        data['Volatility_20d'] = data['Close'].pct_change().rolling(window=20).std()
        
        #Volume
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        
        print("All technical indicators calculated successfully")
        return data
    
    ############ debug #################
    def test_indicators(self, symbol):
        """ Test the calculations"""
        try:
            stock = yf.Ticker(symbol)
            hist_data = stock.history(period='1y')
            
            
            if len(hist_data) < 150:
                return f"Not enough data for {symbol}"
            
            hist_data = self.calculate_technical_indicators(hist_data)
            
            latest = hist_data.iloc[-1]
            
            result = {'symbol': symbol,
            'current_price': round(latest['Close'], 2),
            
            # Trend indicators
            'sma_20': round(latest['SMA_20'], 2),
            'sma_50': round(latest['SMA_50'], 2),
            'price_to_sma20': round(latest['Price_to_SMA20'], 3),
            'price_to_sma50': round(latest['Price_to_SMA50'], 3),
            
            # Momentum
            'rsi': round(latest['RSI'], 1),
            'momentum_1m_pct': round(latest['Momentum_1M'] * 100, 1),
            'momentum_3m_pct': round(latest['Momentum_3M'] * 100, 1),
            'momentum_6m_pct': round(latest['Momentum_6M'] * 100, 1),
            
            # Volatility
            'bb_position': round(latest['BB_Position'], 3),
            'volatility_20d': round(latest['Volatility_20d'], 4),
            
            # Volume
            'volume_ratio': round(latest['Volume_Ratio'], 2),
            
            # Analysis
            'analysis': self._analyze_indicators(latest)
            }
            
            return result
    
        except Exception as e:
            return f"Error: {e}"
        
    def _analyze_indicators(self, latest_data):
        """Create some insight from the indicators"""
        analysis = []
        
        if latest_data['Price_to_SMA20'] > 1.05:
            analysis.append("Strong uptrend (price 5%+ above 20-day average)")
        elif latest_data['Price_to_SMA20'] < 0.95:
            analysis.append("Strong downtrend (price 5%+ below 20-day average)")
        else:
            analysis.append("Sideways trend (price near 20-day average)")
            
        if latest_data['RSI'] > 70:
            analysis.append("Overbought territory (RSI > 70)")
        elif latest_data['RSI'] < 30:
            analysis.append("Oversold territory (RSI < 30)")
        else:
            analysis.append(f"Neutral momentum (RSI: {latest_data['RSI']:.1f})")
            
        if latest_data['BB_Position'] > 0.8:
            analysis.append("Near upper Bollinger Band (high volatility)")
        elif latest_data['BB_Position'] < 0.2:
            analysis.append("Near lower Bollinger Band (potential bounce)")
        else:
            analysis.append("Within normal Bollinger Band range")
            
        if latest_data['Momentum_1M'] > 0.1:
            analysis.append("Strong 1-month momentum (+10%+)")
        elif latest_data['Momentum_1M'] < -0.1:
            analysis.append("Weak 1-month momentum (-10%+)")
            
        if latest_data['Volume_Ratio'] > 1.5:
            analysis.append("High volume (50%+ above average)")
        elif latest_data['Volume_Ratio'] < 0.5:
            analysis.append("Low volume (50%+ below average)")
            
            
        return analysis
stocks = load_stocks()
ml_predictor = SimpleMLPredictor()



@app.route('/')
def index():
    """Main page - shows the stock tracker"""
    global stocks
    
    # Auto-refresh stock data if we have stocks
    if stocks:
        updated_stocks = []
        for stock in stocks:
            fresh_data = get_stock_data(stock['symbol'])
            if fresh_data['valid']:
                updated_stocks.append(fresh_data)
            else:
                updated_stocks.append(stock)  # Keep old data if API fails
        stocks = updated_stocks
    
    return render_template('index.html', stocks=stocks)

@app.route('/add_stock', methods=['POST'])
def add_stock():
    symbol = request.form.get('symbol', '').upper().strip()
    
    if not symbol:
        flash("Please enter a stock symbol", 'error')
        return redirect(url_for('index'))
    
    # Check if already exists
    if symbol in [stock['symbol'] for stock in stocks]:
        flash(f"'{symbol}' is already in your watchlist", 'warning')
        return redirect(url_for('index'))
    
    # Get real stock data
    stock_data = get_stock_data(symbol)
    
    if stock_data['valid']:
        stocks.append(stock_data)
        save_stocks(stocks)
        flash(f"'{symbol}' added successfully!", 'success')
        
    else:
        flash(f"Error: {stock_data['error']}", 'error')
    
    return redirect(url_for('index'))

@app.route('/remove_stock/<symbol>')
def remove_stock(symbol):
    """Remove a stock from the watchlist"""
    global stocks
    stocks = [stock for stock in stocks if stock['symbol'] != symbol]
    save_stocks(stocks)
    flash(f"'{symbol}' removed from watchlist", 'info')
    return redirect(url_for('index'))

@app.route('/refresh_prices')
def refresh_prices():
    """Refreshes all stock prices"""
    global stocks
    updated_stocks = []
    
    for stock in stocks:
        fresh_data = get_stock_data(stock['symbol'])
        
        if fresh_data['valid']:
            updated_stocks.append(fresh_data)
        
        else:
            updated_stocks.append(stock) 
            flash(f"Could not refresh {stocks['symbol']}", 'warning')
            
    stocks = updated_stocks
    save_stocks(stocks)   
    flash("Stock prices refreshed!", 'success')

    return redirect(url_for('index'))

@app.route('/test_indicators/<symbol>')
def test_indicators(symbol):
    """Test"""
    result = ml_predictor.test_indicators(symbol.upper())
    return jsonify(result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
    
