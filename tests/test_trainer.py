"""Tests for DMC training loop."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from zsy.trainer import DMCTrainer, TrainingConfig


class TestDMCTrainer:
    def test_construction(self):
        trainer = DMCTrainer()
        assert trainer.network is not None
        assert trainer.optimizer is not None

    def test_short_training(self):
        """Run a very short training to verify the loop works end-to-end."""
        config = TrainingConfig(
            num_players=3,
            episodes_per_iteration=4,
            num_iterations=2,
            batch_size=32,
            log_every=1,
            checkpoint_every=100,  # don't checkpoint during short test
        )
        trainer = DMCTrainer(config)
        stats = trainer.train()
        assert stats.iteration == 2
        assert stats.total_episodes == 8
        assert stats.total_transitions > 0
        assert len(stats.losses) == 2

    def test_loss_is_finite(self):
        config = TrainingConfig(
            num_players=3,
            episodes_per_iteration=4,
            num_iterations=3,
            batch_size=64,
            log_every=1,
            checkpoint_every=100,
        )
        trainer = DMCTrainer(config)
        stats = trainer.train()
        for loss in stats.losses:
            assert np.isfinite(loss), f"Non-finite loss: {loss}"

    def test_checkpoint_save_load(self):
        config = TrainingConfig(
            num_players=3,
            episodes_per_iteration=4,
            num_iterations=2,
            batch_size=32,
            checkpoint_every=100,
        )
        trainer = DMCTrainer(config)
        trainer.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pt"
            trainer.save_checkpoint(path)
            assert path.exists()

            # Load into a new trainer
            trainer2 = DMCTrainer(config)
            trainer2.load_checkpoint(path)
            assert trainer2.stats.iteration == trainer.stats.iteration

            # Verify network weights match
            for p1, p2 in zip(
                trainer.network.parameters(),
                trainer2.network.parameters(),
            ):
                torch.testing.assert_close(p1, p2)

    def test_gradient_clipping(self):
        """Verify training doesn't produce exploding gradients."""
        config = TrainingConfig(
            num_players=3,
            episodes_per_iteration=8,
            num_iterations=3,
            batch_size=32,
            grad_clip=1.0,
        )
        trainer = DMCTrainer(config)
        stats = trainer.train()
        # If grad clipping works, losses should stay bounded
        assert all(abs(l) < 100 for l in stats.losses)
