"""Tests for legal move generator."""
from zsy.cards import Card, Rank, Suit
from zsy.combinations import Combination, CombinationType, classify
from zsy.legal_moves import legal_moves


def c(rank_str: str, suit_str: str = "h") -> Card:
    rank_map = {
        "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE, "6": Rank.SIX,
        "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE, "10": Rank.TEN,
        "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
        "2": Rank.TWO,
    }
    suit_map = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
    return Card(rank_map[rank_str], suit_map[suit_str])


HR = Rank.TWO


class TestLeadingMoves:
    def test_pass_always_included(self):
        hand = [c("3")]
        moves = legal_moves(hand, HR)
        pass_moves = [m for m in moves if m.combination.type == CombinationType.PASS]
        assert len(pass_moves) == 1

    def test_single_card_hand(self):
        hand = [c("A", "s")]
        moves = legal_moves(hand, HR)
        non_pass = [m for m in moves if m.combination.type != CombinationType.PASS]
        assert len(non_pass) == 1
        assert non_pass[0].combination.type == CombinationType.SINGLE

    def test_pair_in_hand(self):
        hand = [c("5", "h"), c("5", "s")]
        moves = legal_moves(hand, HR)
        types = {m.combination.type for m in moves}
        assert CombinationType.SINGLE in types
        assert CombinationType.PAIR in types

    def test_bomb_in_hand(self):
        hand = [c("8", "h"), c("8", "d"), c("8", "c"), c("8", "s")]
        moves = legal_moves(hand, HR)
        types = {m.combination.type for m in moves}
        assert CombinationType.BOMB in types

    def test_straight_in_hand(self):
        hand = [c("3", "h"), c("4", "d"), c("5", "c"), c("6", "s"), c("7", "h")]
        moves = legal_moves(hand, HR)
        types = {m.combination.type for m in moves}
        assert CombinationType.STRAIGHT in types

    def test_empty_hand(self):
        moves = legal_moves([], HR)
        assert len(moves) == 1
        assert moves[0].combination.type == CombinationType.PASS


class TestFollowingMoves:
    def test_must_beat_single(self):
        hand = [c("3", "h"), c("5", "s"), c("A", "d")]
        active = classify([c("4", "h")])
        moves = legal_moves(hand, HR, active)
        non_pass = [m for m in moves if m.combination.type != CombinationType.PASS]
        # Can play 5 or A, not 3
        ranks = {m.combination.primary_rank for m in non_pass}
        assert Rank.THREE not in ranks
        assert Rank.FIVE in ranks
        assert Rank.ACE in ranks

    def test_must_beat_pair(self):
        hand = [c("3", "h"), c("3", "d"), c("A", "h"), c("A", "s")]
        active = classify([c("5", "h"), c("5", "s")])
        moves = legal_moves(hand, HR, active)
        non_pass = [m for m in moves if m.combination.type != CombinationType.PASS]
        pairs = [m for m in non_pass if m.combination.type == CombinationType.PAIR]
        # Only AA pair beats 55 pair
        assert all(m.combination.primary_rank == Rank.ACE for m in pairs)

    def test_bomb_beats_any_non_bomb(self):
        hand = [c("3", "h"), c("8", "h"), c("8", "d"), c("8", "c"), c("8", "s")]
        active = classify([c("A", "s")])  # A single A
        moves = legal_moves(hand, HR, active)
        non_pass = [m for m in moves if m.combination.type != CombinationType.PASS]
        types = {m.combination.type for m in non_pass}
        assert CombinationType.BOMB in types

    def test_cannot_play_wrong_type(self):
        hand = [c("A", "h"), c("A", "d")]  # Only a pair
        active = classify([c("3", "h")])  # Single
        moves = legal_moves(hand, HR, active)
        non_pass = [m for m in moves if m.combination.type != CombinationType.PASS]
        # Should find single A's, not pairs
        for m in non_pass:
            assert m.combination.type in (CombinationType.SINGLE, CombinationType.BOMB)

    def test_no_valid_beats(self):
        hand = [c("3", "h"), c("4", "d")]
        active = classify([c("A", "s")])
        moves = legal_moves(hand, HR, active)
        non_pass = [m for m in moves if m.combination.type != CombinationType.PASS]
        assert len(non_pass) == 0  # Can only pass

    def test_pass_always_legal_when_following(self):
        hand = [c("3", "h")]
        active = classify([c("A", "s")])
        moves = legal_moves(hand, HR, active)
        pass_moves = [m for m in moves if m.combination.type == CombinationType.PASS]
        assert len(pass_moves) == 1


class TestWildcardInLegalMoves:
    def test_wildcard_expands_options(self):
        wc = Card(Rank.TWO, Suit.HEARTS)  # wildcard when HR=TWO
        hand = [c("5", "s"), wc]
        moves = legal_moves(hand, HR)
        types = {m.combination.type for m in moves}
        # Wildcard + 5♠ can form a pair
        assert CombinationType.PAIR in types

    def test_wildcard_completes_bomb_to_beat(self):
        wc = Card(Rank.TWO, Suit.HEARTS)
        hand = [c("8", "h"), c("8", "d"), c("8", "c"), wc]
        active = classify([c("A", "s")])
        moves = legal_moves(hand, HR, active)
        bombs = [m for m in moves if m.combination.type == CombinationType.BOMB]
        assert len(bombs) >= 1
