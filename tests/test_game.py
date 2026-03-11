"""Tests for game loop and trick resolution."""
import random
from zsy.cards import Card, Rank, Suit
from zsy.combinations import Combination, CombinationType, classify
from zsy.game import Game, GamePhase, GameState, PlayRecord, Agent
from zsy.player import Player
from zsy.wildcard import WildcardAssignment


class RandomAgent:
    """Agent that plays a random legal move."""
    def choose_move(self, player, game_state, moves):
        # Prefer non-pass if possible
        non_pass = [m for m in moves if m.combination.type != CombinationType.PASS]
        if non_pass:
            return random.choice(non_pass)
        return moves[0]  # pass


class AlwaysPassAgent:
    """Agent that always passes (except when leading)."""
    def choose_move(self, player, game_state, moves):
        non_pass = [m for m in moves if m.combination.type != CombinationType.PASS]
        if game_state.active_combo is None or game_state.active_combo.type == CombinationType.PASS:
            # Must play when leading
            if non_pass:
                return non_pass[0]
        return moves[0]  # pass


class SmallestFirstAgent:
    """Agent that always plays the smallest legal move."""
    def choose_move(self, player, game_state, moves):
        non_pass = [m for m in moves if m.combination.type != CombinationType.PASS]
        if not non_pass:
            return moves[0]
        # Play the one with lowest primary rank
        return min(non_pass, key=lambda m: (m.combination.primary_rank or Rank.THREE))


class TestGameSetup:
    def test_game_creation(self):
        agents = [RandomAgent() for _ in range(3)]
        game = Game(3, agents)
        assert game.num_players == 3
        assert game.phase == GamePhase.DEALING

    def test_deal(self):
        agents = [RandomAgent() for _ in range(3)]
        game = Game(3, agents)
        game.deal()
        assert game.phase == GamePhase.PLAYING
        assert all(p.card_count == 18 for p in game.players)

    def test_deal_double_deck(self):
        agents = [RandomAgent() for _ in range(4)]
        game = Game(4, agents)
        game.deal()
        total = sum(p.card_count for p in game.players)
        assert total == 108

    def test_auto_deck_selection(self):
        # 3 players → 1 deck
        game = Game(3, [RandomAgent()] * 3)
        assert game.num_decks == 1
        # 4 players → 2 decks
        game = Game(4, [RandomAgent()] * 4)
        assert game.num_decks == 2


class TestTrickResolution:
    def test_play_single(self):
        agents = [SmallestFirstAgent() for _ in range(3)]
        game = Game(3, agents)
        game.deal()
        record = game.play_turn()
        assert record is not None
        assert record.combination.type != CombinationType.PASS

    def test_pass_increments_counter(self):
        agents = [AlwaysPassAgent() for _ in range(3)]
        game = Game(3, agents)
        game.deal()
        # First player leads (must play something)
        game.play_turn()
        assert game.consecutive_passes == 0
        # Next player passes
        game.play_turn()
        assert game.consecutive_passes == 1

    def test_trick_resets_after_all_pass(self):
        agents = [AlwaysPassAgent() for _ in range(3)]
        game = Game(3, agents)
        game.deal()
        # Player 0 leads
        game.play_turn()
        leader = game.trick_leader
        # Players 1 and 2 pass
        game.play_turn()
        game.play_turn()
        # Trick should reset
        assert game.active_combo is None
        assert game.consecutive_passes == 0


class TestGameCompletion:
    def test_game_runs_to_completion(self):
        random.seed(42)
        agents = [RandomAgent() for _ in range(3)]
        game = Game(3, agents)
        winner = game.run()
        assert game.phase == GamePhase.FINISHED
        assert winner is not None
        assert 0 <= winner < 3
        assert len(game.finish_order) == 3

    def test_multiple_games_produce_winners(self):
        """Run several games to verify stability."""
        for seed in range(5):
            random.seed(seed)
            agents = [RandomAgent() for _ in range(3)]
            game = Game(3, agents)
            winner = game.run()
            assert game.phase == GamePhase.FINISHED
            assert winner is not None

    def test_four_player_game(self):
        random.seed(123)
        agents = [RandomAgent() for _ in range(4)]
        game = Game(4, agents)
        winner = game.run()
        assert game.phase == GamePhase.FINISHED
        assert 0 <= winner < 4

    def test_two_player_game(self):
        random.seed(99)
        agents = [RandomAgent() for _ in range(2)]
        game = Game(2, agents)
        winner = game.run()
        assert game.phase == GamePhase.FINISHED
        assert winner in (0, 1)


class TestGameState:
    def test_state_snapshot(self):
        agents = [RandomAgent() for _ in range(3)]
        game = Game(3, agents)
        game.deal()
        state = game.get_state()
        assert state.current_player == 0
        assert len(state.cards_remaining) == 3
        assert all(cr == 18 for cr in state.cards_remaining)

    def test_play_history_grows(self):
        agents = [SmallestFirstAgent() for _ in range(3)]
        game = Game(3, agents)
        game.deal()
        game.play_turn()
        game.play_turn()
        state = game.get_state()
        assert len(state.play_history) == 2


class TestEdgeCases:
    def test_winner_skipped_in_turns(self):
        """A player who finishes should be skipped in subsequent turns."""
        random.seed(42)
        agents = [RandomAgent() for _ in range(3)]
        game = Game(3, agents)
        winner = game.run()
        # Winner should have 0 cards
        assert game.players[winner].card_count == 0
        # All players should be in finish order
        assert set(game.finish_order) == {0, 1, 2}
