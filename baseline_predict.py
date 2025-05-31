import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# 載入資料
df = pd.read_csv("all_stocks_5yr.csv")
df = df[df['Name'] == 'AAPL'].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.reset_index(drop=True)

# 設定 sequence 長度為 450 天
sequence_length = 450

# 建立序列資料 (X: close 過去450天, y: 第451天 close)
X_seq, y_seq, date_seq = [], [], []
for i in range(len(df) - sequence_length - 1):
    seq = df['close'].iloc[i:i + sequence_length].values
    target = df['close'].iloc[i + sequence_length]
    date = df['date'].iloc[i + sequence_length]
    X_seq.append(seq)
    y_seq.append(target)
    date_seq.append(date)
X = np.array(X_seq)
y = np.array(y_seq)
dates = np.array(date_seq)

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 手動測試集切分（2017/02/01 ~ 2018/02/28）
test_start = pd.to_datetime("2017-02-01")
test_end = pd.to_datetime("2018-02-28")
test_mask = (dates >= test_start) & (dates <= test_end)

X_train = X_scaled[~test_mask]
y_train = y[~test_mask]
X_test = X_scaled[test_mask]
y_test = y[test_mask]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²:  {r2_score(y_test, y_pred):.4f}")

# 視覺化結果
plt.figure(figsize=(10, 5))
plt.plot(dates[test_mask], y_test, label='Actual', linewidth=2)

