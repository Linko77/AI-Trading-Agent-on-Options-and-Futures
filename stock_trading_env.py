import gym
from gym import spaces
import numpy as np
import torch
from multi_stock_preprocessing_7day import MultiStockDatasetPreparer
from lstm_7day import LSTM

class StockTradingEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        file_paths,
        feature_cols,
        lstm_weights_path: str,
        window_size: int = 30,
        horizon: int = 7,
        test_ratio: float = 0.2,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        device: torch.device = None
    ):
        super().__init__()
        # Prepare dataset
        self.preparer = MultiStockDatasetPreparer(
            file_paths=file_paths,
            feature_cols=feature_cols,
            target_col='close',
            window_size=window_size,
            horizon=horizon,
            test_ratio=test_ratio
        )
        X_train, y_train, X_test, y_test = self.preparer.load_and_preprocess()
        # Combine train and test dataset
        self.features = np.concatenate([X_train, X_test], axis=0)
        self.targets = np.concatenate([y_train, y_test], axis=0)
        self.num_samples = self.features.shape[0]

        # LSTM predictor (outputs scaled values)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = LSTM(
            input_size=self.features.shape[2],
            hidden_size=128,
            output_size=horizon,
            seq_len=window_size,
            dropout=0.3
        ).to(self.device)
        self.lstm.load_state_dict(torch.load(lstm_weights_path, map_location=self.device))
        self.lstm.eval()

        # Environment parameters
        self.window_size = window_size
        self.horizon = horizon
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.idx = 0
        # Spaces
        obs_dim = window_size * self.features.shape[2] + horizon
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self):
        # Randomly pick a sample for a new episode
        idx = np.random.randint(0, self.num_samples)
        self.current_window = self.features[idx]  
        self.future_prices_scaled = self.targets[idx]    
        self.balance = self.initial_balance
        self.position = 0
        self.idx = idx
        return self._get_observation()

    def _get_observation(self):
        idx = self.idx
        self.current_window = self.features[idx]  
        self.future_prices_scaled = self.targets[idx]    
        x_tensor = torch.tensor(self.current_window, dtype=torch.float32)
        x_tensor = x_tensor.unsqueeze(0).to(self.device)  
        with torch.no_grad():
            pred_scaled = self.lstm(x_tensor).squeeze(0).cpu().numpy()  
        
        obs = np.concatenate([self.current_window.flatten(), pred_scaled.astype(np.float32)])
        return obs

    def step(self, action: int, time=0):
        # Determine real last price from scaled data
        close_idx = self.preparer.feature_cols.index('close')
        last_price_scaled = self.current_window[-1, close_idx]
        last_price = float(self.preparer.inverse_transform_close(np.array([last_price_scaled]))[0])

        # Convert future scaled prices to real prices
        future_prices_real = self.preparer.inverse_transform_close(self.future_prices_scaled)
        future_prices_real = future_prices_real[-1]

        quantity = 100
        premium = last_price * 0.01
        if action == 1:  # buy call
            profit = max(future_prices_real - last_price, 0) - premium
        elif action == 2:  # buy put
            profit = max(last_price - future_prices_real, 0) - premium
        else:  # hold
            profit = 0
            reward = 0

        # Apply quantity and transaction cost on trade actions (if needed)
        if action in [1, 2]:
            cost = self.transaction_cost * last_price
            reward = (profit - cost)*quantity
            self.balance += reward

        
        if time == 1000:
            done = True  
        else:
            done = False
        self.idx += 1
        if self.idx >= self.num_samples:
            self.idx = 0
        
        obs = self._get_observation()
        info = {'balance': self.balance, 'position': action, 'cost': 0 if action==0 else premium*quantity + cost}
        return obs, reward, done, info

    def render(self, mode='human'):
        print(f"Balance: {self.balance:.2f}")
