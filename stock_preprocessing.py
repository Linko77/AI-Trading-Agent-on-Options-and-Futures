import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockDatasetPreparer:
    def __init__(self, file_path, feature_cols, target_col='close', window_size=60, test_ratio=0.2):
        self.file_path = file_path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window_size = window_size
        self.test_ratio = test_ratio
        self.scaler = MinMaxScaler()

        self.df = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None

    def compute_technical_indicators(self, df):
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()

        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd'] = macd - signal

        return df

    def load_and_preprocess(self):
        df = pd.read_csv(self.file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = self.compute_technical_indicators(df)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.feature_cols = self.feature_cols + ['ma5', 'ma10', 'rsi', 'macd']
        data = df[self.feature_cols].values
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i - self.window_size:i])
            y.append(scaled_data[i, self.feature_cols.index(self.target_col)])


        X, y = np.array(X), np.array(y)
        split_idx = int(len(X) * (1 - self.test_ratio))
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        self.df = df
        return self.X_train, self.y_train, self.X_test, self.y_test

    def inverse_transform_close(self, y_scaled):
        dummy = np.zeros((len(y_scaled), len(self.feature_cols)))
        dummy[:, self.feature_cols.index(self.target_col)] = y_scaled
        inv = self.scaler.inverse_transform(dummy)
        return inv[:, self.feature_cols.index(self.target_col)]
