# Stock Tracker Flask App

A simple web application built with Flask to track your favorite stocks with real-time market data. This is my first project to really attempt to increase my developing and coding skills. 
I hope to gain experience with fullstack development priciples and languages as well as providing some analysis. I hope you like my project!A simple web application built with Flask to track your favorite stocks with real-time market data.

## Features
- Add stocks to your personal watchlist
- View current stock prices and daily changes
- Clean, responsive web interface
- Real-time data from stock APIs
- Built with Python Flask backend

## Tech Stack
- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, Jinja2 templates
- **API**: Stock market data integration
- **Storage**: In-memory (can be extended to database)

## Getting Started

### Prerequisites
- Python 3.7+
- pip (Python package manager)
- Internet connection for stock data

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/bluelion1999/stock-tracker-flask.git
   cd stock-tracker-flask
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and go to `http://localhost:5000`

### Usage
1. Enter a stock symbol (e.g., AAPL, GOOGL, TSLA) in the input field
2. Click "Add Stock" to add it to your watchlist
3. View real-time price data and daily changes
4. Remove stocks from your watchlist as needed

## Project Structure
```
stock-tracker-flask/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # Main HTML template
├── static/
│   └── style.css          # CSS styling
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## API Configuration
The app uses free stock market APIs. You may need to:
1. Sign up for a free API key (Alpha Vantage, Yahoo Finance, etc.)
2. Add your API key to environment variables
3. Update the API endpoints in `app.py`

## Development

### Adding New Features
- Stock data is stored in the `stocks` list (in-memory)
- Add new routes in `app.py` for additional functionality
- Extend templates in the `templates/` folder
- Add styling in `static/style.css`

### Future Enhancements
- [ ] Database integration (SQLite/PostgreSQL)
- [ ] User authentication and multiple watchlists
- [ ] Portfolio value tracking
- [ ] Price alerts and notifications
- [ ] Historical charts and data visualization
- [ ] Dark mode toggle
- [ ] Export watchlist functionality
- [ ] Mobile responsive design improvements

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Flask documentation and community
- Stock market data providers
