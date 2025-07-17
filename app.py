from flask import Flask, render_template, request, redirect, url_for
import requests

# Create Flask app instance
app = Flask(__name__)

# Store stocks in memory for now (like a simple list in Python)
stocks = []

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