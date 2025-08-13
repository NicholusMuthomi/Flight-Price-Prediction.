# Flight Price Prediction.

## Overview

This project develops a machine learning system to predict domestic flight prices in India using a Random Forest regressor trained on historical booking data. The solution includes:
- Comprehensive EDA of flight pricing patterns
- Feature engineering and preprocessing pipeline
- Model training with performance evaluation
- Streamlit web application for interactive price predictions

Achievements:
- Achieved **98.5% R² score** and **₹1,091 MAE** on test data
- Identified key pricing drivers (travel class, airline, duration)
- Developed a production-ready dashboard with price comparison visualizations

## Features

### Core Analysis
- **Data Collection**: 300,153 flight records from Kaggle with 11 features
- **Feature Engineering**: Time-based encoding, route combinations, categorical transformations
- **Exploratory Analysis**: Uncovered price patterns by airline, class, stops and timing

### Machine Learning
- **Model Selection**: Compared 5 algorithms (Random Forest, XGBoost, etc.)
- **Random Forest**: Best performance with 300 trees and optimized hyperparameters
- **Production Pipeline**: Saved model and encoders for deployment

### Visualization
- Interactive Plotly charts comparing airlines
- Glassmorphism UI with dark theme
- Mobile-responsive Streamlit dashboard

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/NicholusMuthomi/flight-price-prediction.git
   cd flight-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure model files are in project root:
   ```bash
   # Required files:
   # - flight_price_prediction_model_random_forest.pkl
   # - flight_price_scaler.pkl (optional)
   # - flight_price_label_encoders.pkl (optional)
   ```

### Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure
```
flight-price-prediction/
├── app.py                     # Streamlit dashboard
├── india_flight_prediction.py  # Model training and analysis
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── flight_price_prediction_model_random_forest.pkl  # Pretrained model
└── assets/                    # Background images
    └── flight.jpg
```

## Results

### Model Performance
| Metric                  | Value       |
|-------------------------|-------------|
| R² Score                | 0.9859      |
| Mean Absolute Error     | ₹1,104      |
| Root Mean Squared Error | ₹2,692      |
| Percentage Accuracy     | 93.24%      |

### Identified Pricing Patterns
1. Business class averages 8x economy prices (₹52,540 vs ₹6,572)
2. Vistara most expensive (₹30,397 avg), AirAsia cheapest (₹4,091)
3. Key price drivers:
   - Travel class (88% feature importance)
   - Flight duration (6%)
   - Days until departure (2%)

## Business Applications

### Travel Industry Use Cases
- **Dynamic Pricing**: Airlines can benchmark fare strategies
- **Travel Agencies**: Build price comparison tools
- **Consumer Savings**: Identify cheapest booking times/routes

### Dashboard Features
- Price prediction for 6 Indian airlines
- Route comparison between 6 major cities
- Visualization of:
  - Price by airline
  - Optimal booking timing
  - Stopover tradeoffs

## Contributing

Contributions are welcome, Please follow these steps:
1. Open an issue to discuss proposed changes
2. Fork the repository and create a feature branch
3. Submit a pull request with:
   - Clear description of changes
   - Updated tests (if applicable)
   - Documentation updates

## License
MIT License. See [LICENSE](LICENSE) for details.

## Contact

For inquiries or collaboration opportunities: muthominicholus22@gmail.com
