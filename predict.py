import torch
import matplotlib.pyplot as plt
from stock_preprocessing import StockDatasetPreparer
from lstm import LSTM
import numpy as np

def predict():
    #set env and load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preparer = StockDatasetPreparer("NRG_data.csv", ['open', 'high', 'low', 'close', 'volume'], window_size=30, test_ratio=0.2)
    X_train, y_train, X_test, y_test = preparer.load_and_preprocess()
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    model = LSTM(input_size=X_train.shape[2], hidden_size=128, output_size=1, seq_len=30).to(device)
    best_model_path = "best_multi_stock_model_2.pth"

    #test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    with torch.no_grad():
        pred = model(X_test.to(device)).squeeze().cpu().numpy()
        true = y_test.squeeze().numpy()

    pred_orig = (preparer.inverse_transform_close(pred))
    true_orig = (preparer.inverse_transform_close(true))

    mse = (np.square(pred_orig - true_orig)).mean()
    r_square = 1 - np.sum(np.square(pred_orig - true_orig)) / np.sum(np.square(true_orig-np.mean(true_orig)))
    print("MSE",mse)
    print("R^2", r_square)
    #show result
    plt.figure(figsize=(10, 5))
    plt.plot(true_orig, label="True Close")
    plt.plot(pred_orig, label="Predicted Close")
    plt.title("NRG Close Price Prediction")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    predict()