from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import yfinance as yf
import json
import os

import pandas as pd
import numpy as np
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
    """Get real data from YF with ML prediction"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Get Current Price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        # Get Previous Close
        previous_close = info.get('previousClose')
        
        if current_price and previous_close:
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100

            stock_data = {
                'symbol': symbol,
                'price': f"{current_price:.2f}",
                'change': f"{change:+.2f} ({change_percent:+.1f}%)",
                'valid': True
            }
            
            # Get ML prediction if model is trained
            if ml_predictor.is_trained:
                ml_result = ml_predictor.predict_stock(symbol)
                if ml_result:
                    stock_data['ml_prediction'] = ml_result
                    print(f"Added ML prediction for {symbol}: {ml_result['recommendation']}")
            
            return stock_data
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
        """Make a buy/sell/hold prediction for a stock"""
        if not self.is_trained:
            print(f"Cannot predict {symbol} - model not trained")
            return None
            
        # Get current features for the stock
        features = self.create_features(symbol)
        if features is None:
            print(f"Cannot get features for {symbol}")
            return None
            
        try:
            # Convert features to array format
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            
            # Scale the features (same way we did in training)
            feature_array_scaled = self.scaler.transform(feature_array)
            
            # Get prediction probabilities
            # [probability_of_not_going_up, probability_of_going_up_5%+]
            probabilities = self.classifier.predict_proba(feature_array_scaled)[0]
            prob_down = probabilities[0]  # Probability of NOT going up 5%+
            prob_up = probabilities[1]    # Probability of going up 5%+
            
            # Get binary prediction (0 or 1)
            binary_prediction = self.classifier.predict(feature_array_scaled)[0]
            
            # Calculate confidence (how sure we are)
            confidence = max(prob_down, prob_up) * 100
            
            print(f"{symbol}: P(up)={prob_up:.3f}, P(down)={prob_down:.3f}, Confidence={confidence:.1f}%")
            
            # Convert to recommendation
            recommendation, signal_strength = self._interpret_prediction(
                binary_prediction, prob_up, confidence, features
            )
            
            return {
                'recommendation': recommendation,
                'confidence': round(confidence, 1),
                'signal_strength': signal_strength,
                'prob_up': round(prob_up * 100, 1),
                'prob_down': round(prob_down * 100, 1),
                'features': features
            }
            
        except Exception as e:
            print(f"Prediction error for {symbol}: {e}")
            return None
    
    
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
    
    def create_features(self, symbol):
        """Create the ML features for a single stock"""
        try:
            stock = yf.Ticker(symbol)
            hist_data = stock.history(period="1y")
            
            if len(hist_data) < 150:
                return f"Insufficient Data to model {symbol}"

            hist_data = self.calculate_technical_indicators(hist_data)
            
            latest = hist_data.iloc[-1]
            
            #feature_dict
            features = {
            'Price_to_SMA20': latest['Price_to_SMA20'],
            'Price_to_SMA50': latest['Price_to_SMA50'],
            'RSI': latest['RSI'],
            'BB_Position': latest['BB_Position'],
            'Momentum_1M': latest['Momentum_1M'],
            'Momentum_3M': latest['Momentum_3M'],
            'Momentum_6M': latest['Momentum_6M'],
            'Volatility_20d': latest['Volatility_20d'],
            'Volume_Ratio': latest['Volume_Ratio']
            }
            
            #clean NaNs
            features = {k: v if not pd.isna(v) else 0 for k, v in features.items()}
            
            return features
        
        except Exception as e:
            print(f"Error creating features for {symbol}: {e}")
            return None
            
    def prepare_training_data(self):
        """Prepare training data for the model"""
        print("Preparing training data")
        
        
        # train ticks
        train_tickers = [
                        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'CRM', 
                        'ORCL', 'ADBE', 'INTC', 'AMD', 'CSCO', 'IBM', 'QCOM', 'AVGO', 'TXN', 'MU', 
                        'AMAT', 'LRCX', 'KLAC', 'ADI', 'MCHP', 'XLNX', 'MRVL', 'SWKS', 'QRVO', 'SLAB', 
                        'CRUS', 'CIRR', 'FORM', 'LITE', 'AMBA', 'SMTC', 'CCMP', 'DIOD', 'MKSI', 'ACLS', 
                        'AEIS', 'COHR', 'VICR', 'POWI', 'MPWR', 'SIMO', 'MTSI', 'TRMB', 'KEYS', 'ANSS', 
                        'CDNS', 'SNPS', 'ADSK', 'CTXS', 'SPLK', 'NOW', 'WDAY', 'VEEV', 'ZS', 'OKTA', 
                        'CRWD', 'NET', 'DDOG', 'SNOW', 'PLTR', 'U', 'RBLX', 'DASH', 'ABNB', 'UBER', 
                        'LYFT', 'TWLO', 'ZM', 'DOCN', 'GTLB', 'MDB', 'TEAM',
                        'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'ABT', 'DHR', 'MRK', 'LLY', 'BMY', 
                        'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'CELG', 'ISRG', 'SYK', 'BSX', 'MDT', 
                        'ZBH', 'EW', 'HOLX', 'VAR', 'ALGN', 'DXCM', 'ILMN', 'IQV', 'PKI', 'A', 
                        'WST', 'MKTX', 'ZTS', 'IDXX', 'RMBS', 'PODD', 'TDOC', 'HALO', 'EDIT', 'CRSP', 
                        'NTLA', 'BLUE', 'SRPT', 'RARE', 'FOLD', 'ARWR', 'BMRN', 'ALNY', 'IONS', 'EXAS', 
                        'NVTA', 'CDNA', 'PACB', 'TWST', 'FATE', 'BEAM', 'PRIME', 'LYEL', 'SEER', 'MYGN',
                        'BRK-A', 'BRK-B', 'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 
                        'MA', 'PYPL', 'SQ', 'COF', 'USB', 'PNC', 'TFC', 'MTB', 'FITB', 'HBAN', 
                        'RF', 'CFG', 'KEY', 'ZION', 'WTFC', 'CMA', 'SIVB', 'ALLY', 'SOFI', 'LC', 
                        'UPST', 'AFRM', 'HOOD', 'COIN', 'MSTR', 'ICE', 'CME', 'NDAQ', 'CBOE', 'SPGI', 
                        'MCO', 'BLK', 'SCHW', 'TROW', 'IVZ', 'BEN', 'STT', 'NTRS', 'AMG', 'WDR',
                        'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'DIS', 'CMCSA', 'CMG', 
                        'YUM', 'QSR', 'ORLY', 'AZO', 'BBY', 'TGT', 'COST', 'WMT', 'KR', 'SYX', 
                        'MHK', 'WHR', 'NVR', 'LEN', 'DHI', 'PHM', 'TOL', 'KBH', 'MTH', 'TPG', 
                        'POOL', 'ULTA', 'EL', 'LULU', 'DECK', 'CROX', 'SKX', 'UAA', 'UA', 'FL', 
                        'JWN', 'M', 'ROST', 'GPS', 'ANF',
                        'PG', 'KO', 'PEP', 'MO', 'PM', 'BTI', 'UL', 'CL', 'KMB', 'GIS', 
                        'K', 'CPB', 'CAG', 'SJM', 'HRL', 'TSN', 'MDLZ', 'MNST', 'KDP', 'STZ', 
                        'DEO', 'SAM', 'TAP', 'BUD', 'HSY', 'MKC', 'SYY', 'ADM', 'BG', 'INGR',
                        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'OXY', 'MPC', 'VLO', 'PSX', 
                        'BKR', 'HAL', 'OIH', 'DVN', 'FANG', 'MRO', 'APA', 'HES', 'NOV', 'RIG', 
                        'HP', 'CHK', 'AR', 'SM', 'WPX', 'MTDR', 'PE', 'QEP', 'NBL', 'COG', 
                        'EQT', 'RRC', 'CNX', 'GPOR', 'WLL',
                        'BA', 'CAT', 'GE', 'HON', 'MMM', 'UPS', 'RTX', 'LMT', 'NOC', 'GD', 
                        'UNP', 'CSX', 'NSC', 'FDX', 'LUV', 'DAL', 'UAL', 'AAL', 'JBLU', 'ALK', 
                        'WM', 'RSG', 'EMR', 'ETN', 'PH', 'ITW', 'DOV', 'ROK', 'CMI', 'DE', 
                        'IR', 'CARR', 'OTIS', 'TDG', 'LDOS', 'HII', 'BWA', 'JCI', 'FAST', 'PCAR',
                        'LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'LYB', 'PPG', 'NEM', 'FCX', 
                        'AA', 'VALE', 'BHP', 'RIO', 'CF', 'MOS', 'IFF', 'FMC', 'ALB', 'SQM', 
                        'VMC', 'MLM', 'CRH', 'UFPI', 'PKG',
                        'AMT', 'CCI', 'PLD', 'EQIX', 'PSA', 'EXR', 'WELL', 'O', 'VTR', 'HCP', 
                        'UDR', 'EQR', 'AVB', 'ESS', 'MAA', 'CPT', 'AIV', 'BXP', 'VNO', 'SLG',
                        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PPL', 'ES', 
                        'ETR', 'CMS', 'DTE', 'NI', 'LNT',
                        'VZ', 'T', 'TMUS', 'CHTR', 'DISH', 'SIRI', 'TWTR', 'SNAP', 'PINS', 'MTCH', 
                        'IAC', 'NWSA', 'NWS', 'FOXA', 'FOX',
                        'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI', 'TME', 'BILI', 'IQ', 
                        'WB', 'NTES', 'TSM', 'ASML', 'SAP', 'NVO', 'AZN', 'RDS-A', 'RDS-B', 'BP', 
                        'TM', 'HMC', 'NSANY', 'SNE', 'NVS', 'ROG', 'NESN', 'LVMUY', 'IDEXY', 'ADDYY',
                        'ROKU', 'DOCU', 'ZI', 'FSLY', 'ESTC', 'FIVN', 'COUP', 'PD', 'PSTG', 'WORK', 
                        'BILL', 'PAYC', 'SHOP', 'PTON', 'BYND', 'SPCE', 'OPEN', 'RDFN', 'Z', 'ZG', 
                        'CARG', 'CVNA', 'CHWY', 'CHEWY', 'ETSY', 'PINS', 'TDOC', 'TELADOC', 'PELOTON', 'ZOOM',
                        'PLUG', 'FCEL', 'BE', 'BLDP', 'NKLA', 'QS', 'LCID', 'RIVN', 'F', 'GM', 
                        'RIDE', 'WKHS', 'HYLN', 'SHLL', 'VLDR', 'LAZR', 'MVIS', 'LIDR', 'OUST', 'AEVA', 
                        'INVZ', 'AEYE', 'GMHI', 'ACTC', 'CCIV'
                    ]
        all_features = []
        all_labels = []
        
        for ticker in train_tickers:
            print(f"Processing {ticker}...")
            
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                hist_data = stock.history(period="2y")  # 2 years for more training data
                
                if len(hist_data) < 200:
                    print(f"Skipping {ticker} - not enough data")
                    continue
                    
                # Calculate indicators
                hist_data = self.calculate_technical_indicators(hist_data)
                
                # Create training examples from historical data
                # We'll look at each day and see if the stock went up 5%+ in the next 20 days
                for i in range(200, len(hist_data) - 20):  # Leave room for future returns
                    current_row = hist_data.iloc[i]
                    future_price = hist_data.iloc[i + 20]['Close']  # Price 20 days later
                    current_price = current_row['Close']
                    
                    # Calculate future return
                    future_return = (future_price - current_price) / current_price
                    
                    # Create features for this day
                    features = {
                        'Price_to_SMA20': current_row['Price_to_SMA20'],
                        'Price_to_SMA50': current_row['Price_to_SMA50'],
                        'RSI': current_row['RSI'],
                        'BB_Position': current_row['BB_Position'],
                        'Momentum_1M': current_row['Momentum_1M'],
                        'Momentum_3M': current_row['Momentum_3M'],
                        'Momentum_6M': current_row['Momentum_6M'],
                        'Volatility_20d': current_row['Volatility_20d'],
                        'Volume_Ratio': current_row['Volume_Ratio']
                    }
                    
                    # Clean features
                    features = {k: v if not pd.isna(v) else 0 for k, v in features.items()}
                    
                    # Only use if all features are valid
                    if all(not pd.isna(v) and abs(v) < 100 for v in features.values()):
                        all_features.append(list(features.values()))
                        # Label: 1 if stock goes up 2%+ in 20 days, 0 otherwise
                        all_labels.append(1 if future_return > 0.02 else 0)
            
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        print(f"Prepared {len(all_features)} training examples")
        return np.array(all_features), np.array(all_labels)
    
    def train_model(self):
        """Train the ML model"""
        try:
            print("Starting model training...")
            
            # Prepare training data
            X, y = self.prepare_training_data()
            
            if len(X) < 100:
                print("Not enough training data")
                return False
            
            print(f"Training on {len(X)} examples")
            print(f"Positive examples (stocks that went up 2%+): {sum(y)}")
            print(f"Negative examples: {len(y) - sum(y)}")
            
            # Initialize ML components
            self.scaler = RobustScaler()  # Scales features to handle outliers
            self.classifier = RandomForestClassifier(
                n_estimators=100,    # 100 decision trees
                random_state=42,     # For reproducible results
                max_depth=10,        # Prevent overfitting
                min_samples_split=20 # Need 20+ samples to split a node
            )
            
            # Scale the features (normalize them)
            X_scaled = self.scaler.fit_transform(X)
            
            # Train the model
            self.classifier.fit(X_scaled, y)
            
            # Test the model's accuracy
            from sklearn.model_selection import cross_val_score
            accuracy_scores = cross_val_score(self.classifier, X_scaled, y, cv=5)
            avg_accuracy = accuracy_scores.mean()
            
            print(f"Model training complete!")
            print(f"Cross-validation accuracy: {avg_accuracy:.1%}")
            
            # Store feature names for later reference
            self.feature_names = [
                'Price_to_SMA20', 'Price_to_SMA50', 'RSI', 'BB_Position',
                'Momentum_1M', 'Momentum_3M', 'Momentum_6M', 'Volatility_20d', 'Volume_Ratio'
            ]
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False
    def _interpret_prediction(self, binary_pred, prob_up, confidence, features):
        """Convert ML output to human-readable recommendation"""
        
        # Only make strong recommendations if we're confident
        if confidence < 60:
            return "HOLD", "Weak Signal"
        
        # Strong buy signals
        if binary_pred == 1 and prob_up > 0.7:
            return "STRONG BUY", "Strong Signal"
        elif binary_pred == 1 and prob_up > 0.6:
            return "BUY", "Moderate Signal"
        
        # Strong sell signals (or avoid buying)
        elif binary_pred == 0 and prob_up < 0.3:
            return "AVOID/SELL", "Strong Signal"
        elif binary_pred == 0 and prob_up < 0.4:
            return "HOLD/SELL", "Moderate Signal"
        
        # Everything else
        else:
            return "HOLD", "Neutral Signal"
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
    
    return render_template('index.html', stocks=stocks,  ml_predictor=ml_predictor)

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

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train the ML model"""
    try:
        success = ml_predictor.train_model()
        
        if success:
            flash("ML model trained successfully! Accuracy information in console.", 'success')
        else:
            flash("Failed to train ML model. Check console for details.", 'error')
    except Exception as e:
        flash(f"Error training model: {str(e)}", 'error')
    
    return redirect(url_for('index'))
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
    
