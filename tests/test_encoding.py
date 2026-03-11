"""Tests for state and action encoding."""
import numpy as np
import pytest

from zsy.cards import Card, Rank, Suit
from zsy.combinations import Combination, CombinationType
from zsy.encoding import (
    CARD_MATRIX_SIZE,
    NUM_RANKS,
    NUM_SUITS,
    MOVE_HISTORY_K,
    card_index,
    cards_to_matrix,
    cards_to_vector,
    encode_action,
    encode_state,
    state_dim,
    action_dim,
)
from zsy.game import GameState, PlayRecord
from zsy.player import Player
from zsy.wildcard import WildcardAssignment


class TestCardIndex:
    def test_standard_card(self):
        card = Card(Rank.THREE, Suit.HEARTS)
        assert card_index(card) == (0, 0)

    def test_two_of_spades(self):
        card = Card(Rank.TWO, Suit.SPADES)
        assert card_index(card) == (12, 3)

    def test_small_joker(self):
        assert card_index(Card.small_joker()) == (13, 0)

    def test_big_joker(self):
        assert card_index(Card.big_joker()) == (14, 0)

    def test_all_standard_ranks_unique(self):
        indices = set()
        for rank in [Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN,
                     Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN,
                     Rank.KING, Rank.ACE, Rank.TWO]:
            for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
                idx = card_index(Card(rank, suit))
                assert idx not in indices, f"Duplicate index {idx}"
                indices.add(idx)
        assert len(indices) == 52


class TestCardsToMatrix:
    def test_empty(self):
        m = cards_to_matrix([])
        assert m.shape == (NUM_RANKS, NUM_SUITS)
        assert m.sum() == 0

    def test_single_card(self):
        m = cards_to_matrix([Card(Rank.ACE, Suit.SPADES)])
        assert m[Rank.ACE.value, 3] == 1.0
        assert m.sum() == 1.0

    def test_pair(self):
        m = cards_to_matrix([
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.KING, Suit.DIAMONDS),
        ])
        assert m[Rank.KING.value, 0] == 1.0
        assert m[Rank.KING.value, 1] == 1.0
        assert m.sum() == 2.0

    def test_double_deck_duplicate(self):
        cards = [Card(Rank.FIVE, Suit.CLUBS), Card(Rank.FIVE, Suit.CLUBS)]
        m = cards_to_matrix(cards)
        assert m[Rank.FIVE.value, 2] == 2.0

    def test_jokers(self):
        m = cards_to_matrix([Card.small_joker(), Card.big_joker()])
        assert m[13, 0] == 1.0
        assert m[14, 0] == 1.0
        assert m.sum() == 2.0


class TestCardsToVector:
    def test_vector_length(self):
        v = cards_to_vector([Card(Rank.THREE, Suit.HEARTS)])
        assert len(v) == CARD_MATRIX_SIZE

    def test_vector_matches_matrix(self):
        cards = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.SPADES)]
        m = cards_to_matrix(cards)
        v = cards_to_vector(cards)
        np.testing.assert_array_equal(v, m.flatten())


class TestEncodeAction:
    def test_pass_is_zero(self):
        wa = WildcardAssignment(Combination.make_pass(), ())
        v = encode_action(wa)
        assert len(v) == CARD_MATRIX_SIZE
        assert v.sum() == 0.0

    def test_single(self):
        card = Card(Rank.SEVEN, Suit.DIAMONDS)
        combo = Combination(CombinationType.SINGLE, (card,), Rank.SEVEN)
        wa = WildcardAssignment(combo, ())
        v = encode_action(wa)
        assert v.sum() == 1.0
        # Check correct position
        idx = Rank.SEVEN.value * NUM_SUITS + 1  # diamonds = suit 1
        assert v[idx] == 1.0

    def test_bomb(self):
        cards = tuple(Card(Rank.NINE, s) for s in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES])
        combo = Combination(CombinationType.BOMB, cards, Rank.NINE, bomb_size=4)
        wa = WildcardAssignment(combo, ())
        v = encode_action(wa)
        assert v.sum() == 4.0

    def test_wildcard_substitution(self):
        # Wildcard substituted as a card - appears at the substituted position
        real_card = Card(Rank.FIVE, Suit.HEARTS)
        sub_card = Card(Rank.FIVE, Suit.CLUBS)
        wild_card = Card(Rank.TWO, Suit.HEARTS)  # the wildcard
        combo = Combination(CombinationType.PAIR, (real_card, sub_card), Rank.FIVE)
        wa = WildcardAssignment(combo, ((wild_card, sub_card),))
        v = encode_action(wa)
        assert v.sum() == 2.0


class TestEncodeState:
    def _make_player(self, pid: int, hand: list[Card]) -> Player:
        p = Player(id=pid)
        p.hand = hand
        return p

    def _make_game_state(
        self, current_player: int = 0, num_players: int = 3
    ) -> GameState:
        return GameState(
            current_player=current_player,
            active_combo=None,
            trick_leader=0,
            consecutive_passes=0,
            cards_remaining=[18] * num_players,
        )

    def test_state_vector_shape(self):
        player = self._make_player(0, [Card(Rank.ACE, Suit.HEARTS)])
        gs = self._make_game_state()
        v = encode_state(player, gs, num_players=3)
        expected_dim = state_dim(3, teams=False)
        assert len(v) == expected_dim

    def test_state_with_teams(self):
        player = self._make_player(0, [Card(Rank.ACE, Suit.HEARTS)])
        gs = self._make_game_state()
        gs.teams = [0, 1, 0]
        v = encode_state(player, gs, num_players=3)
        expected_dim = state_dim(3, teams=True)
        assert len(v) == expected_dim

    def test_hand_encoding_in_state(self):
        hand = [Card(Rank.THREE, Suit.HEARTS), Card(Rank.KING, Suit.SPADES)]
        player = self._make_player(0, hand)
        gs = self._make_game_state()
        v = encode_state(player, gs, num_players=3)
        # First 60 values should be the hand encoding
        hand_vec = cards_to_vector(hand)
        np.testing.assert_array_equal(v[:CARD_MATRIX_SIZE], hand_vec)

    def test_move_history_encoding(self):
        player = self._make_player(0, [])
        gs = self._make_game_state()
        card = Card(Rank.ACE, Suit.HEARTS)
        combo = Combination(CombinationType.SINGLE, (card,), Rank.ACE)
        gs.play_history = [PlayRecord(player_id=1, combination=combo, cards_played=(card,))]
        v = encode_state(player, gs, num_players=3)
        # Should not crash and have correct length
        assert len(v) == state_dim(3)

    def test_wildcard_flag(self):
        # Player with high_card_rank=TWO holding 2♥ (wildcard)
        wc = Card(Rank.TWO, Suit.HEARTS)
        player = self._make_player(0, [wc])
        player.high_card_rank = Rank.TWO
        gs = self._make_game_state()
        v = encode_state(player, gs, num_players=3)
        # Wildcard flag is near the end
        expected_dim = state_dim(3)
        # wildcard flag is at position: hand + others_played + history + remaining + ranks
        offset = (
            CARD_MATRIX_SIZE  # hand
            + 2 * CARD_MATRIX_SIZE  # 2 other players
            + MOVE_HISTORY_K * (CARD_MATRIX_SIZE + 3)  # history
            + 3  # remaining
            + 3 * 13  # rank one-hots
        )
        assert v[offset] == 1.0

    def test_bomb_count(self):
        player = self._make_player(0, [])
        gs = self._make_game_state()
        # Add a bomb to history
        cards = tuple(Card(Rank.NINE, s) for s in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES])
        combo = Combination(CombinationType.BOMB, cards, Rank.NINE, bomb_size=4)
        gs.play_history = [PlayRecord(player_id=1, combination=combo, cards_played=cards)]
        v = encode_state(player, gs, num_players=3)
        # Bomb count is after wildcard flag
        offset = (
            CARD_MATRIX_SIZE
            + 2 * CARD_MATRIX_SIZE
            + MOVE_HISTORY_K * (CARD_MATRIX_SIZE + 3)
            + 3
            + 3 * 13
            + 1  # wildcard flag
        )
        assert v[offset] == 1.0


class TestDimensions:
    def test_state_dim_3_players(self):
        d = state_dim(3)
        assert d > 0
        # hand(60) + 2*played(120) + 15*(60+3)(945) + 3 + 3*13(39) + 1 + 1
        expected = 60 + 120 + 945 + 3 + 39 + 1 + 1
        assert d == expected

    def test_state_dim_with_teams(self):
        d_no = state_dim(3, teams=False)
        d_yes = state_dim(3, teams=True)
        assert d_yes == d_no + 3

    def test_action_dim(self):
        assert action_dim() == CARD_MATRIX_SIZE
        assert action_dim() == 60
