"""Meta-learning based ParamController for Differential Evolution.

Provides `DiscreteDQNController` implementing the `ParamController` interface.
The controller wraps a small PyTorch network that maps observations to
discrete DE parameter values (F, CR). It can load/save a model
and be used at inference time inside the DE optimizer.

Training: online DQN training with experience replay.
Inference: pure greedy decision-making (epsilon=0).
"""
from typing import Dict, Any
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from param_controller import ParamController

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED, set_global_seed


class _Net(nn.Module):
    """Three-layer fully connected feedforward neural network as the Q-network.
    
    Architecture: 3 -> 128 -> 128 -> 128 -> 25
    All weights initialized via Kaiming uniform, biases initialized to zero.
    """
    def __init__(self, obs_dim, action_dim, seed=None):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)
        
        if seed is not None:
            self._init_weights(seed)

    def _init_weights(self, seed):
        """使用固定种子初始化网络权重"""
        g = torch.Generator()
        g.manual_seed(seed)
        
        for m in [self.fc1, self.fc2, self.fc3, self.out]:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.type_as(self.fc1.weight)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return self.out(x)


class DiscreteDQNController(ParamController):
    """Discrete-action DQN controller for DE parameters with online training.

    Discretizes F and CR into K=5 equally spaced values and
    treats the Cartesian product as the discrete action space (|A|=25).
    The network outputs Q-values for each discrete action; `step`
    returns the greedy action as a dict (matching ParamController).

    Online training: stores transitions in a replay buffer and performs
    a DQN update every `update_freq` steps.
    
    Action space (K=5, |A|=25):
        F ∈ {0.1, 0.325, 0.55, 0.775, 1.0}
        CR ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
    
    Training hyperparameters (per paper):
        - Replay buffer capacity: 1000
        - Batch size: 64
        - Update frequency: 10 steps
        - Target update frequency: 50 steps
        - Discount factor gamma: 0.99
        - Optimizer: Adam, lr=1e-4
        - Gradient clipping: max_norm=1.0
        - Epsilon-greedy: start=1.0, end=0.05, decay=0.999
    """
    def __init__(self, model_path: str = None, obs_dim: int = 3, bins: int = 5,
                device: str = 'cpu', training: bool = False, update_freq: int = 10,
                buffer_capacity: int = 1000, batch_size: int = 64,
                lr: float = 1e-4, gamma: float = 0.99, target_update_freq: int = 50,
                max_grad_norm: float = 1.0, epsilon_start: float = 1.0,
                epsilon_end: float = 0.05, epsilon_decay: float = 0.999,
                seed: int = None):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.bins = int(bins)
        self.seed = seed if seed is not None else 42
        
        # 设置PyTorch种子确保网络初始化可复现
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        # 确保PyTorch操作完全确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        
        self.rng = np.random.RandomState(self.seed)

        # DE parameter discretization (K=5, |A|=25)
        self.F_vals = np.linspace(0.1, 1.0, self.bins)
        self.CR_vals = np.linspace(0.0, 1.0, self.bins)

        self.actions = []
        for fi in self.F_vals:
            for cri in self.CR_vals:
                self.actions.append({'F': float(fi), 'CR': float(cri)})
        self.n_actions = len(self.actions)  # 25
        
        # 使用种子初始化网络
        self.net = _Net(self.obs_dim, self.n_actions, seed=self.seed).to(self.device)

        # Training setup
        self.training = training
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.step_count = 0
        self.total_loss = 0.0
        self.total_reward = 0.0
        self.loss_history = []
        self.reward_history = []
        self.return_record_interval = 5
        self.current_sweep_reward = 0.0

        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        if training:
            self.target_net = _Net(self.obs_dim, self.n_actions, seed=self.seed).to(self.device)
            self.target_net.load_state_dict(self.net.state_dict())
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
            self.buffer_capacity = buffer_capacity
            self.replay_buffer = []
            self.buffer_pos = 0

        if model_path and os.path.exists(model_path):
            try:
                self.load(model_path)
            except Exception:
                pass

    def reset(self, problem_meta: Dict[str, Any]) -> None:
        return None

    def _obs_to_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Convert observation dict to tensor.
        
        State vector s_t = [p_t, rho_t, tilde{f}_t]
        - p_t: normalized optimization progress
        - rho_t: relative fitness improvement using log(1+|f|)
        - tilde{f}_t: normalized best fitness using tanh(log10(1+|f|)*sign(f)/10)
        """
        evals = float(observation.get('evals', 0))
        gbest_f = float(observation.get('gbest_f', 0.0))
        init_f = float(observation.get('initial_gbest_f', gbest_f))
        max_evals = float(observation.get('max_evals', max(1.0, evals)))
        
        # Normalized optimization progress
        progress = min(1.0, evals / max(1.0, max_evals))
        
        # Relative fitness improvement: log(1+|f|) formulation
        if init_f != 0.0 and gbest_f != 0.0 and np.isfinite(init_f) and np.isfinite(gbest_f):
            rel_improve = max(0.0, min(1.0, 
                (np.log(1 + abs(init_f)) - np.log(1 + abs(gbest_f))) / max(1.0, np.log(1 + abs(init_f)))
            ))
        else:
            rel_improve = 0.0
        
        # Normalized best fitness: tanh(log10(1+|f|)*sign(f)/10)
        if np.isfinite(gbest_f) and gbest_f != 0.0:
            gbest_f_norm = float(np.tanh(np.log10(1 + abs(gbest_f)) * np.sign(gbest_f) / 10.0))
        else:
            gbest_f_norm = 0.0
        
        obs = [progress, rel_improve, gbest_f_norm]
        net_dtype = self.net.fc1.weight.dtype
        return torch.tensor(obs, dtype=net_dtype, device=self.device)

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        x = self._obs_to_tensor(observation)

        # Epsilon-greedy action selection (only in training mode)
        if self.training and self.rng.random() < self.epsilon:
            idx = int(self.rng.randint(self.n_actions))
        else:
            with torch.no_grad():
                q = self.net(x.unsqueeze(0)).squeeze(0).cpu().numpy()
            idx = int(np.argmax(q))

        if self.training:
            # Store transition placeholder; reward will be set later via store_reward()
            self._last_obs = observation
            self._last_action_idx = idx
            self._last_obs_tensor = x

        return dict(self.actions[idx])

    def store_reward(self, reward: float, next_observation: Dict[str, Any], done: bool):
        """Store a complete transition (obs, action, reward, next_obs, done) in the replay buffer.
        
        Binary reward function:
            R_t = 1 if f_t < f_{t-1} (improved)
            R_t = 0 otherwise
        """
        # Always track total reward (for both train and test)
        self.total_reward += reward
        self.current_sweep_reward += reward
        
        if not self.training:
            return

        next_obs_tensor = self._obs_to_tensor(next_observation)
        transition = (
            self._last_obs_tensor.clone(),
            self._last_action_idx,
            reward,
            next_obs_tensor.clone(),
            done
        )

        if len(self.replay_buffer) < self.buffer_capacity:
            self.replay_buffer.append(transition)
        else:
            self.replay_buffer[self.buffer_pos] = transition
            self.buffer_pos = (self.buffer_pos + 1) % self.buffer_capacity

        self.step_count += 1

        # Record return every N updates and reset counter
        if self.step_count % self.return_record_interval == 0:
            self.reward_history.append(self.current_sweep_reward)
            self.current_sweep_reward = 0.0

        # Train every update_freq steps
        if self.step_count % self.update_freq == 0 and len(self.replay_buffer) >= self.batch_size:
            loss = self._train_step()
            self.total_loss += loss
            self.loss_history.append(loss)

    def _train_step(self) -> float:
        """Perform one DQN training step on a batch from the replay buffer.
        
        Minimizes TD loss: L(theta) = E[(Q_theta(s,a) - y)^2]
        where y = R + gamma * (1 - done) * max_a' Q_theta-(s', a')
        """
        batch = self.rng.choice(len(self.replay_buffer), self.batch_size, replace=False).tolist()
        batch = [self.replay_buffer[i] for i in batch]

        obs_b = torch.stack([t[0] for t in batch])
        act_b = torch.tensor([t[1] for t in batch], dtype=torch.long, device=self.device)
        rew_b = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=self.device)
        next_b = torch.stack([t[3] for t in batch])
        done_b = torch.tensor([t[4] for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.net(obs_b)
        q_sa = q_values.gather(1, act_b.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(next_b)
            q_next_max = q_next.max(1)[0]
            target = rew_b + self.gamma * (1.0 - done_b) * q_next_max

        loss = F.mse_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Update target network periodically
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def reset_loss(self):
        """Reset loss and reward counters for a new sweep."""
        self.total_loss = 0.0
        self.total_reward = 0.0
        self.current_sweep_reward = 0.0
        self.reward_history = []

    def get_loss_history(self) -> list:
        """Return the list of recorded losses."""
        return self.loss_history

    def get_reward_history(self) -> list:
        """Return the list of recorded returns (every N updates)."""
        return self.reward_history

    def get_total_loss(self) -> float:
        """Return the cumulative training loss."""
        return self.total_loss

    def get_total_reward(self) -> float:
        """Return the cumulative reward from environment interactions."""
        return self.total_reward

    def action_to_index(self, action: Dict[str, float]) -> int:
        a = np.array([action.get('F', 0.8), action.get('CR', 0.9)], dtype=float)
        acts = np.array([[aa['F'], aa['CR']] for aa in self.actions], dtype=float)
        d = np.sum((acts - a[None, :])**2, axis=1)
        return int(np.argmin(d))

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))