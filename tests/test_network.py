"""Tests for Q-network architecture."""
import numpy as np
import torch
import pytest

from zsy.encoding import state_dim, action_dim, CARD_MATRIX_SIZE
from zsy.network import QNetwork


class TestQNetwork:
    def test_construction(self):
        net = QNetwork(num_players=3)
        assert isinstance(net, torch.nn.Module)

    def test_forward_shape(self):
        net = QNetwork(num_players=3)
        batch = 4
        s_dim = state_dim(3)
        a_dim = action_dim()
        state = torch.randn(batch, s_dim)
        action = torch.randn(batch, a_dim)
        q = net(state, action)
        assert q.shape == (batch, 1)

    def test_forward_with_teams(self):
        net = QNetwork(num_players=4, teams=True)
        batch = 2
        s_dim = state_dim(4, teams=True)
        a_dim = action_dim()
        state = torch.randn(batch, s_dim)
        action = torch.randn(batch, a_dim)
        q = net(state, action)
        assert q.shape == (batch, 1)

    def test_forward_single_sample(self):
        net = QNetwork(num_players=3)
        s_dim = state_dim(3)
        a_dim = action_dim()
        state = torch.randn(1, s_dim)
        action = torch.randn(1, a_dim)
        q = net(state, action)
        assert q.shape == (1, 1)

    def test_best_action(self):
        net = QNetwork(num_players=3)
        s_dim = state_dim(3)
        a_dim = action_dim()
        state = np.random.randn(s_dim).astype(np.float32)
        actions = [np.random.randn(a_dim).astype(np.float32) for _ in range(5)]
        idx = net.best_action(state, actions)
        assert 0 <= idx < 5

    def test_best_action_epsilon_1(self):
        """With epsilon=1, should always be random."""
        net = QNetwork(num_players=3)
        s_dim = state_dim(3)
        a_dim = action_dim()
        state = np.random.randn(s_dim).astype(np.float32)
        actions = [np.random.randn(a_dim).astype(np.float32) for _ in range(10)]
        # Run many times, should get variety
        indices = {net.best_action(state, actions, epsilon=1.0) for _ in range(50)}
        assert len(indices) > 1  # Should be random

    def test_gradients_flow(self):
        """Verify gradients flow through the network."""
        net = QNetwork(num_players=3)
        s_dim = state_dim(3)
        a_dim = action_dim()
        state = torch.randn(2, s_dim)
        action = torch.randn(2, a_dim)
        q = net(state, action)
        loss = q.sum()
        loss.backward()
        # Check at least some params have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
        assert has_grad

    def test_parameter_count_reasonable(self):
        """Network shouldn't be too small or too large."""
        net = QNetwork(num_players=3)
        total_params = sum(p.numel() for p in net.parameters())
        # Should have substantial parameters (MLP with 512 hidden × 6 layers)
        assert total_params > 100_000
        assert total_params < 10_000_000

    def test_different_player_counts(self):
        """Networks for different player counts should have different input dims."""
        net3 = QNetwork(num_players=3)
        net4 = QNetwork(num_players=4)
        assert net3._static_dim != net4._static_dim
