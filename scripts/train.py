"""Train a DouZero-style RL agent for Zheng Shang You.

Usage:
    python scripts/train.py [--iterations N] [--episodes-per-iter N] [--eval-every N]

This script trains a Q-network using Deep Monte Carlo self-play,
periodically evaluating against Random and Heuristic baselines.
"""
import argparse
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zsy.trainer import DMCTrainer, TrainingConfig
from zsy.evaluate import evaluate_vs_baselines
from zsy.agents import RandomAgent, HeuristicAgent
from zsy.self_play import RLAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def make_rl_agent_factory(trainer):
    """Create a factory that produces RL agents using the trainer's network."""
    def factory():
        return RLAgent(trainer.network, num_players=3, epsilon=0.0)
    return factory


def main():
    parser = argparse.ArgumentParser(description="Train ZSY RL agent")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Number of training iterations")
    parser.add_argument("--episodes-per-iter", type=int, default=32,
                        help="Episodes per training iteration")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate every N iterations")
    parser.add_argument("--eval-games", type=int, default=200,
                        help="Games per evaluation")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for model checkpoints")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Exploration rate")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size")
    args = parser.parse_args()

    config = TrainingConfig(
        num_players=3,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        episodes_per_iteration=args.episodes_per_iter,
        num_iterations=args.iterations,
        epsilon=args.epsilon,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.eval_every,
        log_every=10,
    )

    trainer = DMCTrainer(config)
    logger.info(f"Starting training: {args.iterations} iterations, "
                f"{args.episodes_per_iter} episodes/iter, lr={args.lr}")

    # Train in chunks, evaluating periodically
    completed = 0
    while completed < args.iterations:
        chunk = min(args.eval_every, args.iterations - completed)
        trainer.train(num_iterations=chunk)
        completed += chunk

        # Evaluate
        logger.info(f"=== Evaluation after {completed} iterations ===")
        result = evaluate_vs_baselines(
            make_rl_agent_factory(trainer),
            agent_name="RL",
            num_games=args.eval_games,
            num_players=3,
        )
        logger.info("\n" + result.summary())

    logger.info("Training complete!")
    logger.info(f"Final model saved to {args.checkpoint_dir}/model_final.pt")


if __name__ == "__main__":
    main()
