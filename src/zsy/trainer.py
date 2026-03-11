"""Deep Monte Carlo (DMC) training loop for DouZero-style RL.

Training procedure:
1. Generate episodes via self-play with epsilon-greedy exploration
2. For each (state, action) pair, use the episode return as the target
3. Update Q-network to minimize MSE: ||Q(s,a) - G||²
4. Periodically checkpoint the model

The return G_t for each step is simply the terminal reward (+1/-1)
since gamma=1 and there are no intermediate rewards (every-visit MC).
"""
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from zsy.network import QNetwork
from zsy.self_play import Episode, Transition, generate_episodes

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for DMC training."""
    num_players: int = 3
    teams: bool = False
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 256
    episodes_per_iteration: int = 64
    num_iterations: int = 1000
    epsilon: float = 0.01
    grad_clip: float = 1.0
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 50
    # Logging
    log_every: int = 10


@dataclass
class TrainingStats:
    """Accumulated training statistics."""
    iteration: int = 0
    total_episodes: int = 0
    total_transitions: int = 0
    losses: list[float] = field(default_factory=list)
    win_rates: list[float] = field(default_factory=list)


class DMCTrainer:
    """Deep Monte Carlo trainer for the Q-network."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.network = QNetwork(
            num_players=self.config.num_players,
            teams=self.config.teams,
        )
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
        )
        self.loss_fn = nn.MSELoss()
        self.stats = TrainingStats()

    def train(self, num_iterations: int | None = None) -> TrainingStats:
        """Run the full training loop.

        Args:
            num_iterations: Override config.num_iterations if provided.

        Returns:
            TrainingStats with accumulated metrics.
        """
        n_iter = num_iterations or self.config.num_iterations
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_iter):
            self.stats.iteration = i + 1

            # Generate episodes
            episodes = generate_episodes(
                self.network,
                num_episodes=self.config.episodes_per_iteration,
                num_players=self.config.num_players,
                epsilon=self.config.epsilon,
            )

            # Collect transitions
            transitions = []
            wins_by_player: dict[int, int] = {}
            for ep in episodes:
                transitions.extend(ep.transitions)
                wins_by_player[ep.winner] = wins_by_player.get(ep.winner, 0) + 1

            self.stats.total_episodes += len(episodes)
            self.stats.total_transitions += len(transitions)

            if not transitions:
                continue

            # Train on collected transitions
            loss = self._train_batch(transitions)
            self.stats.losses.append(loss)

            # Win rate of player 0 as a proxy metric
            wr = wins_by_player.get(0, 0) / len(episodes)
            self.stats.win_rates.append(wr)

            # Logging
            if (i + 1) % self.config.log_every == 0:
                avg_loss = np.mean(self.stats.losses[-self.config.log_every:])
                avg_wr = np.mean(self.stats.win_rates[-self.config.log_every:])
                logger.info(
                    f"Iter {i+1}/{n_iter} | Loss: {avg_loss:.4f} | "
                    f"WR(p0): {avg_wr:.3f} | Episodes: {self.stats.total_episodes} | "
                    f"Transitions: {self.stats.total_transitions}"
                )

            # Checkpoint
            if (i + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(checkpoint_dir / f"model_{i+1}.pt")

        # Final checkpoint
        self.save_checkpoint(checkpoint_dir / "model_final.pt")
        return self.stats

    def _train_batch(self, transitions: list[Transition]) -> float:
        """Train on a batch of transitions. Returns average loss."""
        self.network.train()

        # Prepare data
        states = np.stack([t.state for t in transitions])
        actions = np.stack([t.action for t in transitions])
        returns = np.array([t.reward for t in transitions], dtype=np.float32)

        states_t = torch.from_numpy(states)
        actions_t = torch.from_numpy(actions)
        returns_t = torch.from_numpy(returns).unsqueeze(1)

        # Mini-batch training
        n = len(transitions)
        indices = np.random.permutation(n)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, n, self.config.batch_size):
            end = min(start + self.config.batch_size, n)
            batch_idx = indices[start:end]

            batch_states = states_t[batch_idx]
            batch_actions = actions_t[batch_idx]
            batch_returns = returns_t[batch_idx]

            # Forward pass
            q_values = self.network(batch_states, batch_actions)
            loss = self.loss_fn(q_values, batch_returns)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "stats": {
                    "iteration": self.stats.iteration,
                    "total_episodes": self.stats.total_episodes,
                    "total_transitions": self.stats.total_transitions,
                },
                "config": {
                    "num_players": self.config.num_players,
                    "teams": self.config.teams,
                },
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        stats = checkpoint.get("stats", {})
        self.stats.iteration = stats.get("iteration", 0)
        self.stats.total_episodes = stats.get("total_episodes", 0)
        self.stats.total_transitions = stats.get("total_transitions", 0)
        logger.info(f"Loaded checkpoint from {path}")
