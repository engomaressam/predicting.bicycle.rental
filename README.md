# Predicting Bicycle Rental - Capstone Project

## Project Overview
This project explores the relationship between weather conditions and bicycle rentals in Washington DC. The goal is to predict the number of bicycles needed on a given day based on weather conditions.

## Dataset
The dataset contains daily weather information including:
- **Temperature** (temp, atemp)
- **Humidity** (hum)
- **Wind speed** (windspeed)
- **Weather situation** (weathersit)
- **Season, year, month**
- **Holiday, weekday, workingday indicators**
- **Bike rental counts** (casual, registered, total count)

## Project Structure
```
predicting.bicycle.rental/
├── data/               # Data files
├── models/            # Saved models
├── notebooks/         # Jupyter notebooks
├── visualizations/    # Generated plots and charts
├── day.csv           # Daily bike rental data
├── hour.csv          # Hourly bike rental data
├── Bikes.ipynb       # Main analysis notebook
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

## Installation
```bash
pip install -r requirements.txt
```

## Project Steps

### Step 1: Data Preparation
- Load and clean the dataset
- Explore missing values and outliers
- Transform data for analysis

### Step 2: Exploratory Data Analysis (EDA)
- Visualize relationships between weather variables and bike rentals
- Compute correlations
- Identify patterns and trends

### Step 3: Feature Engineering
- Create new features
- Engineer weather patterns over multiple days
- Create interaction features

### Step 4: Model Development
- Split data into training and testing sets
- Implement multiple machine learning algorithms:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Train and evaluate models

### Step 5: Model Refinement
- Hyperparameter tuning
- Cross-validation
- Evaluation metrics (MSE, RMSE, R², MAE)

### Step 6: Conclusion
- Compare model performance
- Discuss strengths and weaknesses
- Provide recommendations

## Usage
Open and run the `Bikes.ipynb` notebook to see the complete analysis.

## Results
Results, visualizations, and model performance metrics are documented in the notebook and saved in the `visualizations/` and `models/` directories.

## Author
Data Science Capstone Project

## License
Educational Project
