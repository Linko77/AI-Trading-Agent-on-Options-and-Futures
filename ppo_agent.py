import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm 
from torch.distributions import Categorical
from stock_trading_env import StockTradingEnv

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value.squeeze(-1)

class PPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr=3e-4,
                 gamma=0.99,
                 clip_epsilon=0.2,
                 update_epochs=10,
                 minibatch_size=64):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Storage buffers
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs, value = self.model(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob.item(), value.item()

    def store_transition(self, state, action, logprob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value=0):
        returns = []
        advantages = []
        G = last_value
        for reward, value, done in zip(reversed(self.rewards), reversed(self.values), reversed(self.dones)):
            G = reward + self.gamma * G * (1 - done)
            adv = G - value
            returns.insert(0, G)
            advantages.insert(0, adv)
        return returns, advantages

    def update(self):
        # Convert to tensors
        
        states = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions = torch.tensor(np.array(self.actions))
        old_logprobs = torch.tensor(self.logprobs, dtype=torch.float32)
        returns, advantages = self.compute_returns_and_advantages()
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        # PPO update loop
        dataset = torch.utils.data.TensorDataset(states, actions, old_logprobs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        for _ in range(self.update_epochs):
            for batch_states, batch_actions, batch_old_logprobs, batch_returns, batch_advantages in loader:
                probs, values = self.model(batch_states)
                dist = Categorical(probs)

                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(values, batch_returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear buffers
        self.states, self.actions, self.logprobs, self.values, self.rewards, self.dones = [], [], [], [], [], []


def train():
    # Hyperparameters
    file_paths = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".csv")]
    feature_cols = ["open", "high", "low", "close", "volume"]
    lstm_weights = "best_multi_stock_7day.pth"
    env = StockTradingEnv(file_paths, feature_cols, lstm_weights)

    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        clip_epsilon=0.2
    )

    max_episodes = 1000
    print_interval = 1
    all_rewards = []

    for ep in tqdm(range(1, max_episodes + 1), desc="Training"):
        state = env.reset()
        done = False
        ep_reward = 0
        time = 0
        actions = [0,0,0]
        
        while not done:
            time += 1
            action, logprob, value = agent.select_action(state)
            actions[action]+=1
            next_state, reward, done, info = env.step(action, time)
            agent.store_transition(state, action, logprob, value, reward, done)
            state = next_state
            ep_reward += reward

        agent.update()
        all_rewards.append(ep_reward)

        if ep % print_interval == 0:
            avg_reward = np.mean(all_rewards[-print_interval:])
            print(f"Episode {ep}\tAvg Reward: {avg_reward:.2f}", info["balance"], actions)

    # Save trained model
    os.makedirs("models", exist_ok=True)
    torch.save(agent.model.state_dict(), "models/ppo_lstm_stock.pth")

if __name__ == '__main__':
    train()
