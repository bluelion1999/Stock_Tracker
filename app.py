from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf

# Create Flask app instance
app = Flask(__name__)

# Store stocks in memory
stocks = []     

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


@app.route('/')
def index():
    """Main page - shows the stock tracker"""
    return render_template('index.html', stocks=stocks)

@app.route('/add_stock', methods=['POST'])
def add_stock():
    """Add a new stock to the watchlist"""
    # Get the stock symbol from the form
    symbol = request.form.get('symbol', '').upper().strip()
    
    # Check if symbol is not empty and not already in list
    if symbol and symbol not in [stock['symbol'] for stock in stocks]:
        # Create stock data (we'll add real API data later)
        stock_data = {
            'symbol': symbol, 
            'price': '0.00', 
            'change': '+0.00'
        }
        stocks.append(stock_data)
    
    # Redirect back to main page
    return redirect(url_for('index'))

@app.route('/remove_stock/<symbol>')
def remove_stock(symbol):
    """Remove a stock from the watchlist"""
    global stocks
    stocks = [stock for stock in stocks if stock['symbol'] != symbol]
    return redirect(url_for('index'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)