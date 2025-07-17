from flask import Flask, render_template, request, redirect, url_for, flash
import yfinance as yf

# Create Flask app instance
app = Flask(__name__)

# Store stocks in memory
stocks = []     

app.secret_key = 'your-secret-key-for-flash-messages'

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
        flash(f"'{symbol}' added successfully!", 'success')
    else:
        flash(f"Error: {stock_data['error']}", 'error')
    
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