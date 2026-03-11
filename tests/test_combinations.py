"""Tests for card combination types and validation."""
import pytest
from zsy.cards import Card, Rank, Suit
from zsy.combinations import (
    Combination,
    CombinationType,
    classify,
)

# Helpers
def c(rank_str: str, suit_str: str = "h") -> Card:
    """Shorthand card constructor. e.g. c("3", "h") = 3 of hearts."""
    rank_map = {
        "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE, "6": Rank.SIX,
        "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE, "10": Rank.TEN,
        "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
        "2": Rank.TWO,
    }
    suit_map = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
    return Card(rank_map[rank_str], suit_map[suit_str])


class TestClassifySingle:
    def test_single(self):
        combo = classify([c("3")])
        assert combo is not None
        assert combo.type == CombinationType.SINGLE
        assert combo.primary_rank == Rank.THREE

    def test_single_joker(self):
        combo = classify([Card.small_joker()])
        assert combo is not None
        assert combo.type == CombinationType.SINGLE


class TestClassifyPair:
    def test_pair(self):
        combo = classify([c("5", "h"), c("5", "s")])
        assert combo is not None
        assert combo.type == CombinationType.PAIR
        assert combo.primary_rank == Rank.FIVE

    def test_pair_different_ranks_invalid(self):
        assert classify([c("5", "h"), c("6", "h")]) is None

    def test_joker_pair_same_type(self):
        combo = classify([Card.small_joker(), Card.small_joker()])
        assert combo is not None
        assert combo.type == CombinationType.PAIR

    def test_joker_pair_mixed_invalid(self):
        assert classify([Card.small_joker(), Card.big_joker()]) is None


class TestClassifyTriple:
    def test_triple(self):
        combo = classify([c("K", "h"), c("K", "d"), c("K", "c")])
        assert combo is not None
        assert combo.type == CombinationType.TRIPLE
        assert combo.primary_rank == Rank.KING


class TestClassifyFullHouse:
    def test_full_house(self):
        cards = [c("K", "h"), c("K", "d"), c("K", "c"), c("3", "h"), c("3", "s")]
        combo = classify(cards)
        assert combo is not None
        assert combo.type == CombinationType.TRIPLE_PLUS_PAIR
        assert combo.primary_rank == Rank.KING

    def test_full_house_pair_rank_is_not_primary(self):
        cards = [c("3", "h"), c("3", "d"), c("3", "c"), c("A", "h"), c("A", "s")]
        combo = classify(cards)
        assert combo is not None
        assert combo.primary_rank == Rank.THREE


class TestClassifyStraight:
    def test_straight(self):
        cards = [c("3", "h"), c("4", "d"), c("5", "c"), c("6", "s"), c("7", "h")]
        combo = classify(cards)
        assert combo is not None
        assert combo.type == CombinationType.STRAIGHT
        assert combo.primary_rank == Rank.SEVEN

    def test_straight_high(self):
        cards = [c("10", "h"), c("J", "d"), c("Q", "c"), c("K", "s"), c("A", "h")]
        combo = classify(cards)
        assert combo is not None
        assert combo.type == CombinationType.STRAIGHT
        assert combo.primary_rank == Rank.ACE

    def test_straight_with_2_invalid(self):
        cards = [c("10", "h"), c("J", "d"), c("Q", "c"), c("K", "s"), c("2", "h")]
        assert classify(cards) is None

    def test_straight_non_consecutive_invalid(self):
        cards = [c("3", "h"), c("4", "d"), c("5", "c"), c("7", "s"), c("8", "h")]
        assert classify(cards) is None

    def test_straight_four_cards_invalid(self):
        cards = [c("3", "h"), c("4", "d"), c("5", "c"), c("6", "s")]
        assert classify(cards) is None

    def test_straight_six_cards_invalid(self):
        cards = [c("3", "h"), c("4", "d"), c("5", "c"), c("6", "s"), c("7", "h"), c("8", "d")]
        assert classify(cards) is None


class TestClassifyStraightFlush:
    def test_straight_flush(self):
        cards = [c("3", "h"), c("4", "h"), c("5", "h"), c("6", "h"), c("7", "h")]
        combo = classify(cards)
        assert combo is not None
        assert combo.type == CombinationType.STRAIGHT_FLUSH
        assert combo.primary_rank == Rank.SEVEN

    def test_straight_flush_over_straight(self):
        """Straight flush should be classified as such, not as a regular straight."""
        cards = [c("8", "s"), c("9", "s"), c("10", "s"), c("J", "s"), c("Q", "s")]
        combo = classify(cards)
        assert combo.type == CombinationType.STRAIGHT_FLUSH


class TestClassifyConsecutivePairs:
    def test_consecutive_pairs(self):
        cards = [c("3", "h"), c("3", "s"), c("4", "h"), c("4", "s"), c("5", "h"), c("5", "s")]
        combo = classify(cards)
        assert combo is not None
        assert combo.type == CombinationType.CONSECUTIVE_PAIRS
        assert combo.primary_rank == Rank.FIVE

    def test_non_consecutive_pairs_invalid(self):
        cards = [c("3", "h"), c("3", "s"), c("5", "h"), c("5", "s"), c("7", "h"), c("7", "s")]
        assert classify(cards) is None


class TestClassifyConsecutiveTriples:
    def test_consecutive_triples(self):
        cards = [
            c("3", "h"), c("3", "d"), c("3", "c"),
            c("4", "h"), c("4", "d"), c("4", "c"),
        ]
        combo = classify(cards)
        assert combo is not None
        assert combo.type == CombinationType.CONSECUTIVE_TRIPLES
        assert combo.primary_rank == Rank.FOUR


class TestClassifyBomb:
    def test_four_of_a_kind(self):
        cards = [c("8", "h"), c("8", "d"), c("8", "c"), c("8", "s")]
        combo = classify(cards)
        assert combo is not None
        assert combo.type == CombinationType.BOMB
        assert combo.bomb_size == 4

    def test_five_of_a_kind(self):
        # Possible with 2 decks
        cards = [c("8", "h"), c("8", "d"), c("8", "c"), c("8", "s"), c("8", "h")]
        combo = classify(cards)
        assert combo is not None
        assert combo.type == CombinationType.BOMB
        assert combo.bomb_size == 5

    def test_six_of_a_kind(self):
        cards = [c("8", s) for s in ("h", "d", "c", "s", "h", "d")]
        combo = classify(cards)
        assert combo is not None
        assert combo.type == CombinationType.BOMB
        assert combo.bomb_size == 6

    def test_eight_of_a_kind(self):
        cards = [c("8", s) for s in ("h", "d", "c", "s", "h", "d", "c", "s")]
        combo = classify(cards)
        assert combo is not None
        assert combo.bomb_size == 8


class TestClassifyPass:
    def test_empty_is_pass(self):
        combo = classify([])
        assert combo is not None
        assert combo.type == CombinationType.PASS

    def test_explicit_pass(self):
        combo = Combination.make_pass()
        assert combo.type == CombinationType.PASS


class TestClassifyInvalid:
    def test_two_different_ranks(self):
        assert classify([c("3", "h"), c("5", "h")]) is None

    def test_four_cards_not_bomb(self):
        cards = [c("3", "h"), c("3", "s"), c("4", "h"), c("4", "s")]
        assert classify(cards) is None

    def test_seven_random_cards(self):
        cards = [c("3", "h"), c("4", "d"), c("5", "c"), c("6", "s"),
                 c("7", "h"), c("9", "d"), c("J", "c")]
        assert classify(cards) is None


class TestBeats:
    def test_higher_single_beats_lower(self):
        a = classify([c("A")])
        b = classify([c("3")])
        assert a.beats(b)
        assert not b.beats(a)

    def test_higher_pair_beats_lower(self):
        a = classify([c("K", "h"), c("K", "s")])
        b = classify([c("Q", "h"), c("Q", "s")])
        assert a.beats(b)

    def test_pair_cannot_beat_single(self):
        a = classify([c("3", "h"), c("3", "s")])
        b = classify([c("A")])
        assert not a.beats(b)

    def test_bomb_beats_non_bomb(self):
        bomb = classify([c("3", "h"), c("3", "d"), c("3", "c"), c("3", "s")])
        single = classify([c("A")])
        assert bomb.beats(single)
        assert not single.beats(bomb)

    def test_bomb_beats_straight(self):
        bomb = classify([c("3", "h"), c("3", "d"), c("3", "c"), c("3", "s")])
        straight = classify([c("10", "h"), c("J", "d"), c("Q", "c"), c("K", "s"), c("A", "h")])
        assert bomb.beats(straight)

    def test_higher_bomb_beats_lower(self):
        high = classify([c("A", "h"), c("A", "d"), c("A", "c"), c("A", "s")])
        low = classify([c("3", "h"), c("3", "d"), c("3", "c"), c("3", "s")])
        assert high.beats(low)
        assert not low.beats(high)

    def test_five_of_kind_beats_four(self):
        five = classify([c("3", s) for s in ("h", "d", "c", "s", "h")])
        four = classify([c("A", "h"), c("A", "d"), c("A", "c"), c("A", "s")])
        assert five.beats(four)
        assert not four.beats(five)

    def test_straight_flush_beats_five_of_kind(self):
        sf = classify([c("3", "h"), c("4", "h"), c("5", "h"), c("6", "h"), c("7", "h")])
        five = classify([c("A", s) for s in ("h", "d", "c", "s", "h")])
        assert sf.beats(five)
        assert not five.beats(sf)

    def test_six_of_kind_beats_straight_flush(self):
        six = classify([c("3", s) for s in ("h", "d", "c", "s", "h", "d")])
        sf = classify([c("10", "s"), c("J", "s"), c("Q", "s"), c("K", "s"), c("A", "s")])
        assert six.beats(sf)
        assert not sf.beats(six)

    def test_pass_never_beats(self):
        p = Combination.make_pass()
        single = classify([c("3")])
        assert not p.beats(single)
        assert single.beats(p)

    def test_pass_does_not_beat_pass(self):
        p1 = Combination.make_pass()
        p2 = Combination.make_pass()
        assert not p1.beats(p2)

    def test_higher_straight_beats_lower(self):
        high = classify([c("4", "h"), c("5", "d"), c("6", "c"), c("7", "s"), c("8", "h")])
        low = classify([c("3", "h"), c("4", "d"), c("5", "c"), c("6", "s"), c("7", "d")])
        assert high.beats(low)
        assert not low.beats(high)

    def test_higher_sf_beats_lower_sf(self):
        high = classify([c("4", "s"), c("5", "s"), c("6", "s"), c("7", "s"), c("8", "s")])
        low = classify([c("3", "h"), c("4", "h"), c("5", "h"), c("6", "h"), c("7", "h")])
        assert high.beats(low)

    def test_is_bomb_property(self):
        bomb = classify([c("3", "h"), c("3", "d"), c("3", "c"), c("3", "s")])
        sf = classify([c("3", "h"), c("4", "h"), c("5", "h"), c("6", "h"), c("7", "h")])
        single = classify([c("3")])
        assert bomb.is_bomb
        assert sf.is_bomb
        assert not single.is_bomb
