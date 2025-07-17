from flask import Flask, render_template, request, redirect, url_for
import requests

#create the flask instance
app = Flask(__name__)

#Store stocks in memory 
#TODO: add real backend db

stocks = []

@app.route('/')
def index():
    """ Main page - shows the stock tracker"""
    return render_template('index.html', stocks=stocks)

@app.route('/add_stock', methods=['POST'])
def add_stock():
    """Add a new stock to the watchlist"""
    #Get the ticker from the form
    symbol = request.form['symbol'].upper().strip()
    
    #Check if symbol is non-empty or already marked
    if symbol and symbol not in [stock['symbol'] for stock in stocks]:
        #Get stock data
        #TODO add api integration to get real data
        stock_data = {
            'symbol' : symbol,
            'price' : '0.00',
            'change' : '+0.00'
        }
        stocks.append(stock_data)
    #route to main        
    return(redirect(url_for('index')))

@app.route('/remove_stock/<symbol>')
def remove_stock(symbol):
    """Remove a stock from the watchlist"""
    global stocks
    stocks = [stock for stock in stocks if stock['symbol'] != symbol]
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
