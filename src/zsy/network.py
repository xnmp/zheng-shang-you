"""Q-network architecture for DouZero-style DMC training.

Action-in architecture: takes (state, action) as input, outputs scalar Q-value.
Only legal actions are evaluated at each step.

Architecture:
- State features (hand, played cards, remaining, ranks, flags) → MLP embedding
- Move history sequence → LSTM → final hidden state
- Concatenate [state_embedding, history_embedding, action_features] → MLP → Q-value
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from zsy.encoding import (
    CARD_MATRIX_SIZE,
    MOVE_HISTORY_K,
    NUM_STANDARD_RANKS,
    state_dim,
    action_dim,
)


class QNetwork(nn.Module):
    """DouZero-style Q-network with LSTM history encoding.

    Input: (state_vector, action_vector)
    Output: scalar Q-value

    The state vector is split into:
    - Static features (hand, played cards, remaining, ranks, flags)
    - History features (last K moves) → fed through LSTM
    """

    def __init__(
        self,
        num_players: int = 3,
        teams: bool = False,
        lstm_hidden: int = 256,
        mlp_hidden: int = 512,
        mlp_layers: int = 6,
    ) -> None:
        super().__init__()
        self.num_players = num_players
        self.teams = teams
        self.lstm_hidden = lstm_hidden

        # Calculate feature dimensions
        self._move_feat_dim = CARD_MATRIX_SIZE + num_players  # per-move features
        self._static_dim = self._calc_static_dim(num_players, teams)

        # LSTM for move history
        self.history_lstm = nn.LSTM(
            input_size=self._move_feat_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
        )

        # Input to MLP: static features + LSTM output + action features
        mlp_input_dim = self._static_dim + lstm_hidden + action_dim()

        # MLP layers
        layers: list[nn.Module] = []
        in_dim = mlp_input_dim
        for _ in range(mlp_layers):
            layers.append(nn.Linear(in_dim, mlp_hidden))
            layers.append(nn.ReLU())
            in_dim = mlp_hidden
        layers.append(nn.Linear(mlp_hidden, 1))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _calc_static_dim(num_players: int, teams: bool) -> int:
        """Dimension of the non-history part of the state vector."""
        dim = CARD_MATRIX_SIZE  # hand
        dim += (num_players - 1) * CARD_MATRIX_SIZE  # others' played cards
        dim += num_players  # cards remaining
        dim += num_players * NUM_STANDARD_RANKS  # high-card ranks
        dim += 1  # wildcard flag
        dim += 1  # bomb count
        if teams:
            dim += num_players  # team flags
        return dim

    def _split_state(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split state vector into static features and history sequence.

        Args:
            state: (batch, state_dim) tensor

        Returns:
            static_feats: (batch, static_dim)
            history_seq: (batch, K, move_feat_dim)
        """
        # Layout in encoding.py:
        # [hand | others_played | history_moves | remaining | ranks | wildcard | bomb | teams]
        # History starts after hand + others_played
        history_start = CARD_MATRIX_SIZE + (self.num_players - 1) * CARD_MATRIX_SIZE
        history_end = history_start + MOVE_HISTORY_K * self._move_feat_dim

        static_before = state[:, :history_start]
        history_flat = state[:, history_start:history_end]
        static_after = state[:, history_end:]

        # Reshape history into sequence
        history_seq = history_flat.view(-1, MOVE_HISTORY_K, self._move_feat_dim)

        # Concatenate non-history static features
        static_feats = torch.cat([static_before, static_after], dim=1)

        return static_feats, history_seq

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Compute Q-value for (state, action) pairs.

        Args:
            state: (batch, state_dim) tensor
            action: (batch, action_dim) tensor

        Returns:
            q_values: (batch, 1) tensor
        """
        static_feats, history_seq = self._split_state(state)

        # LSTM encoding of move history
        lstm_out, _ = self.history_lstm(history_seq)
        history_embedding = lstm_out[:, -1, :]  # last hidden state

        # Concatenate all features
        combined = torch.cat([static_feats, history_embedding, action], dim=1)

        return self.mlp(combined)

    def best_action(
        self,
        state: np.ndarray,
        actions: list[np.ndarray],
        epsilon: float = 0.0,
    ) -> int:
        """Select the best action index from a list of legal actions.

        Args:
            state: State vector (1D numpy array)
            actions: List of action vectors (1D numpy arrays)
            epsilon: Probability of random action (for exploration)

        Returns:
            Index into the actions list
        """
        if np.random.random() < epsilon:
            return np.random.randint(len(actions))

        self.eval()
        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0)
            # Batch all actions together for efficiency
            state_batch = state_t.expand(len(actions), -1)
            action_batch = torch.from_numpy(np.stack(actions))
            q_values = self.forward(state_batch, action_batch)
            return int(q_values.argmax().item())
