# House Price Prediction ML

An end-to-end machine learning project for predicting house prices using real estate data. Includes data exploration, feature engineering, model training, and a FastAPI-based REST API for production predictions.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Features](#features)

## 🎯 Project Overview

This project demonstrates a complete machine learning pipeline from raw real estate data to a deployed prediction API. It includes:

- **Exploratory Data Analysis (EDA)**: Statistical analysis and visualization of housing data
- **Feature Engineering**: Automated preprocessing and feature transformation
- **Model Training**: XGBoost-based price prediction with hyperparameter optimization
- **API Deployment**: FastAPI REST API for real-time predictions

## 📁 Project Structure

```
house-price-prediction-ml/
├── api/                          # FastAPI application
│   ├── app.py                   # Main API endpoints
│   └── requirements.txt          # API dependencies
├── data/
│   ├── raw/                     # Original unprocessed data
│   └── processed/               # Cleaned and transformed data
├── models/                       # Trained model artifacts
│   └── house_price_model_bundle.pkl  # Serialized pipeline + model
├── notebooks/
│   ├── eda.ipynb                # Exploratory data analysis
│   └── train.ipynb              # Model training notebook
├── src/
│   ├── preprocessing/
│   │   └── feature_pipeline.py  # Feature engineering pipeline
│   ├── training/                # Training scripts
│   ├── inference/               # Inference utilities
│   └── utils/                   # Helper functions
├── reports/                      # Analysis outputs and visualizations
│   └── eda/                     # EDA reports and statistics
├── LICENSE
└── README.md

```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd house-price-prediction-ml
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate      # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   # For API only
   cd api
   pip install -r requirements.txt
   
   # Or for full development
   pip install fastapi uvicorn pandas numpy scikit-learn joblib xgboost jupyter
   ```

## 📖 Usage

### Running the API

Start the FastAPI server with automatic reload:

```bash
cd api
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```bash
GET /health
```

Returns the API status.

#### Predict House Price
```bash
POST /predict
```

**Request body:**
```json
{
  "size": 2500,
  "bedrooms": 4,
  "bathrooms": 3,
  "year_built": 2010,
  "condition": "Good",
  "type": "House",
  "location": "Downtown",
  "date_sold": "2024-01-15"
}
```

**Response:**
```json
{
  "predicted_price": 450000.50
}
```

### Using the Interactive API Docs

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to test endpoints directly in your browser.

### Training the Model

To train a new model using the notebooks:

1. **Open the training notebook**
   ```bash
   jupyter notebook notebooks/train.ipynb
   ```

2. **Run exploratory analysis**
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

## 🔌 API Documentation

### Input Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `size` | float | Property size in square feet | 2500 |
| `bedrooms` | int | Number of bedrooms | 4 |
| `bathrooms` | int | Number of bathrooms | 3 |
| `year_built` | int | Year the property was built | 2010 |
| `condition` | string | Property condition | "Good", "Excellent", "Fair" |
| `type` | string | Property type | "House", "Condo", "Townhouse" |
| `location` | string | Location/City | "Downtown", "Suburbs" |
| `date_sold` | string | Date sold (YYYY-MM-DD) | "2024-01-15" |

### Example Requests

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "size": 2500,
    "bedrooms": 4,
    "bathrooms": 3,
    "year_built": 2010,
    "condition": "Good",
    "type": "House",
    "location": "Downtown",
    "date_sold": "2024-01-15"
  }'
```

**Using Python:**
```python
import requests

data = {
    "size": 2500,
    "bedrooms": 4,
    "bathrooms": 3,
    "year_built": 2010,
    "condition": "Good",
    "type": "House",
    "location": "Downtown",
    "date_sold": "2024-01-15"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## 🤖 Model Details

### Architecture
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Feature Engineering**: Custom pipeline with:
  - Temporal feature extraction (year, month from dates)
  - Ordinal encoding for condition
  - One-hot encoding for categorical features
  - Automatic imputation for missing values

### Training Process
1. Data preprocessing and feature engineering
2. Train-test split
3. Hyperparameter tuning via grid search
4. Model serialization with joblib
5. Bundling with feature pipeline for reproducibility

### Model Artifacts
- **Location**: `models/house_price_model_bundle.pkl`
- **Contains**: Feature engineering pipeline + trained XGBoost model
- **Size**: Bundled for easy deployment

## ✨ Features

### Preprocessing Pipeline
- **Automatic imputation**: Median imputation for numeric, mode for categorical
- **Feature creation**: House age calculation from year built
- **Date decomposition**: Extract temporal features from sale dates
- **Categorical encoding**: Ordinal and one-hot encoding with intelligent handling
- **Dropping identifiers**: Automatic removal of non-predictive ID columns

### API Features
- **Input validation**: Pydantic models ensure data integrity
- **Descriptive documentation**: All parameters documented in API docs
- **Error handling**: Clear error messages for invalid inputs
- **CORS ready**: Can be extended for cross-origin requests
- **Scalable**: Production-ready with Uvicorn ASGI server

## 📊 Reports

Analysis outputs are stored in `reports/eda/`:
- `summary.txt`: Statistical summaries
- `pivot_*.csv`: Cross-tabulation analyses
- `price_by_*.csv`: Price distributions by features
- `vif.csv`: Variance Inflation Factor analysis

## 🔧 Technologies Used

- **FastAPI**: Web framework for building APIs
- **Uvicorn**: ASGI server
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: ML preprocessing and utilities
- **XGBoost**: Gradient boosting machine learning model
- **Joblib**: Model serialization
- **Jupyter**: Interactive notebooks for analysis

## 📝 License

See LICENSE file for details.

## 🤝 Contributing

To contribute to this project:
1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 💬 Support

For issues, questions, or suggestions, please open an issue on the repository.
