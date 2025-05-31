import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import joblib

# === USER CONFIGURATION ===
NUM_STOCKS_TO_USE = 500  # Set how many stocks to train on
RANDOM_SEED = 42

SAVE_MODELS = True               # Toggle this to True/False
TRANNING = True
# ===========================

# Load dataset
df = pd.read_csv("archive/all_stocks_5yr.csv")
#tickers_to_use = ["NRG"]

#Get all unique tickers
if NUM_STOCKS_TO_USE==500:
    tickers_to_use = sorted(df['Name'].unique())
else:
    tickers = sorted(df['Name'].unique())
    np.random.seed(RANDOM_SEED)
    tickers_to_use = np.random.choice(tickers, NUM_STOCKS_TO_USE, replace=False)

# Store evaluation results
results = []
all_preds = []
all_targets = []

# Define feature generation function
def generate_features(data):
    data['return_1d'] = data['close'].pct_change()
    data['ma_5'] = data['close'].rolling(window=5).mean()
    data['ma_10'] = data['close'].rolling(window=10).mean()
    data['volatility_5'] = data['return_1d'].rolling(window=5).std()
    data['target'] = data['close'].shift(-1)

    #new add feature
    # data['ma_ratio'] = data['ma_5'] / data['ma_10']
    # data['price_change'] = data['close'] - data['open']
    # data['high_low_range'] = data['high'] - data['low']

    return data


# Train model for each selected stock
for ticker in tqdm(tickers_to_use, desc=f"Training on {NUM_STOCKS_TO_USE} stocks"):
    stock_df = df[df['Name'] == ticker].copy()
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df.sort_values('date', inplace=True)
    
    stock_df = generate_features(stock_df)
    stock_df.dropna(inplace=True)
    
    if len(stock_df) < 100:  # Skip if not enough data
        continue

    split_index = int(0.8 * len(stock_df))
    train = stock_df[:split_index]
    test = stock_df[split_index:]
    
    features = ['open', 'high', 'low', 'volume', 'ma_5', 'ma_10', 'volatility_5']
    #features = ['open', 'high', 'low', 'volume', 'ma_5', 'ma_10', 'volatility_5', 'ma_ratio', 'price_change', 'high_low_range']

    
    if TRANNING:
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
        model.fit(train[features], train['target'])
        if SAVE_MODELS:
            joblib.dump(model, f"rf_models/rf_{ticker}.joblib")
    else:
        loaded_model = joblib.load(f"rf_models/rf_{ticker}.joblib")
        model = loaded_model
          
    preds = model.predict(test[features])
    
    all_preds.extend(preds)
    all_targets.extend(test['target'].values)

    mse = mean_squared_error(test['target'], preds)
    r2 = r2_score(test['target'], preds)
    
    results.append({
        'Ticker': ticker,
        'MSE': mse,
        'R2': r2
    })


# Results summary
overall_r2 = r2_score(all_targets, all_preds)
results_df = pd.DataFrame(results)
print(f"\nOverall R² across all {NUM_STOCKS_TO_USE} stocks: {overall_r2:.4f}")
summary = results_df.sort_values('R2', ascending=False)
print("\nTop 5 models by R²:")
print(summary.head(5))
print("\nLast 5 models by R²:")
print(summary.tail(5))
# print(results_df[results_df["Ticker"] == "AAPL"])

# Optional: plot one stock result
sample_stock = np.random.choice(results_df['Ticker'].values)  # random
#sample_stock = results_df.sort_values('R2').iloc[502]['Ticker'] #highest R-square
#sample_stock = results_df.sort_values('R2').iloc[0]['Ticker'] #lowest R-square
#sample_stock = "AAPL"
sample_df = df[df['Name'] == sample_stock].copy()
sample_df['date'] = pd.to_datetime(sample_df['date'])
sample_df.sort_values('date', inplace=True)
sample_df = generate_features(sample_df).dropna()
split = int(0.8 * len(sample_df))
test = sample_df[split:]
preds = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED).fit(
    sample_df[:split][features], sample_df[:split]['target']
).predict(test[features])

plt.figure(figsize=(12, 6))
plt.plot(test['date'], test['target'], label='Actual')
plt.plot(test['date'], preds, label='Predicted')
plt.title(f'{sample_stock} Close Price Prediction (Random Forest)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()
