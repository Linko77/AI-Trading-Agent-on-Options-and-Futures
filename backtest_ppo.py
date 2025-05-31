import os
import torch
import numpy as np
from stock_trading_env import StockTradingEnv
from multi_stock_preprocessing_7day import MultiStockDatasetPreparer
from ppo_agent import PPOAgent
import matplotlib.pyplot as plt

def backtest():
    # File path
    file_paths = ["NRG_data.csv"]
    feature_cols = ["open", "high", "low", "close", "volume"]
    lstm_weights = "best_multi_stock_7day.pth"
    ppo_weights = "models/ppo_lstm_stock.pth"

    # Prepare data
    preparer = MultiStockDatasetPreparer(
        file_paths=file_paths,
        feature_cols=feature_cols,
        target_col='close',
        window_size=30,
        horizon=7,
        test_ratio=0.2
    )
    X_train, y_train, X_test, y_test = preparer.load_and_preprocess()

    # Initialize environment (for inverse scaling and prediction)
    env = StockTradingEnv(
        file_paths=file_paths,
        feature_cols=feature_cols,
        lstm_weights_path=lstm_weights,
        window_size=30,
        horizon=7,
        test_ratio=0.2,
        initial_balance=100000,
        transaction_cost=0
    )
    env.features = X_test
    env.targets = y_test
    env.num_samples = X_test.shape[0]
    env.balance = env.initial_balance
    env.idx = 0

    # Load trained PPO agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)
    agent.model.load_state_dict(torch.load(ppo_weights, map_location="cpu"))
    agent.model.eval()

    balances = []
    actions = []
    rewards = []
    costs = []
    for idx in range(env.num_samples):
        # Set environment to sample
        env.current_window = env.features[idx]
        env.future_prices_scaled = env.targets[idx]
        
        obs = env._get_observation()
        action, _, _ = agent.select_action(obs)
        # Step
        _, reward, done, info = env.step(action)

        balances.append(info['balance']+reward)
        actions.append(action)
        rewards.append(reward)
        costs.append(info['cost'])


    #DCA
    dca_principle = 100000
    dca_cash = 0
    dca_position = 0
    dca_value = []
    interval = 5
    amount_per_buy = 200

    true_orig = preparer.inverse_transform_close(y_test[:,0])
    for i in range(1, len(true_orig)):
        if i % interval == 0:
            buy_price = true_orig[i]
            shares_bought = amount_per_buy / buy_price
            dca_position += shares_bought
            dca_cash += amount_per_buy
        value = dca_position * true_orig[i] + dca_principle - dca_cash
        dca_value.append(value)


    # Result
    initial = env.initial_balance
    final = env.balance
    total_cost = np.sum(costs)
    total_return = (final - initial) / total_cost * 100
    avg_reward = np.mean(rewards)
    print("PPO+LSTM")
    print("actions:",actions.count(0),actions.count(1),actions.count(2))
    print(f"Initial Balance: {initial:.2f}")
    print(f"Final Balance:   {final:.2f}")
    print(f"Total cost:    {total_cost:.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Average Reward:  {avg_reward:.2f}")

    dca_final = dca_value[-1]
    dca_return = (dca_final - dca_principle) / (dca_cash) * 100
    print("DCA")
    print(f"Initial Balance: {dca_principle:.2f}")
    print(f"Final Balance:   {dca_final:.2f}")
    print(f"Total cost:    {dca_cash:.2f}")
    print(f"Total Return:    {dca_return:.2f}%")
    
    plt.figure(figsize=(10, 5))
    plt.plot(balances, label="PPO+LSTM")
    plt.plot(dca_value, label="DCA")
    plt.title("Backtesting: PPO+LSTM vs DCA")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    backtest()
