"""Tests for wildcard mechanics."""
import pytest
from zsy.cards import Card, Rank, Suit
from zsy.combinations import CombinationType
from zsy.wildcard import classify_with_wildcards, WildcardAssignment


def wc() -> Card:
    """The default wildcard: 2 of hearts (when high card rank is 2)."""
    return Card(Rank.TWO, Suit.HEARTS)


def c(rank_str: str, suit_str: str = "h") -> Card:
    rank_map = {
        "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE, "6": Rank.SIX,
        "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE, "10": Rank.TEN,
        "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
        "2": Rank.TWO,
    }
    suit_map = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
    return Card(rank_map[rank_str], suit_map[suit_str])


HR = Rank.TWO  # default high card rank


class TestNoWildcards:
    def test_regular_pair(self):
        results = classify_with_wildcards([c("5", "h"), c("5", "s")], HR)
        assert len(results) == 1
        assert results[0].combination.type == CombinationType.PAIR
        assert results[0].assignments == ()

    def test_invalid_cards(self):
        results = classify_with_wildcards([c("5", "h"), c("6", "s")], HR)
        assert len(results) == 0


class TestWildcardPair:
    def test_wildcard_completes_pair(self):
        # 5♠ + wildcard → can form pair of 5s
        results = classify_with_wildcards([c("5", "s"), wc()], HR)
        types = {r.combination.type for r in results}
        assert CombinationType.PAIR in types
        # Find the pair of 5s
        pair_5 = [r for r in results
                  if r.combination.type == CombinationType.PAIR
                  and r.combination.primary_rank == Rank.FIVE]
        assert len(pair_5) == 1
        assert len(pair_5[0].assignments) == 1


class TestWildcardTriple:
    def test_wildcard_completes_triple(self):
        results = classify_with_wildcards([c("K", "h"), c("K", "d"), wc()], HR)
        types = {r.combination.type for r in results}
        assert CombinationType.TRIPLE in types
        triple_k = [r for r in results
                    if r.combination.type == CombinationType.TRIPLE
                    and r.combination.primary_rank == Rank.KING]
        assert len(triple_k) == 1


class TestWildcardBomb:
    def test_wildcard_completes_four_of_a_kind(self):
        results = classify_with_wildcards(
            [c("8", "h"), c("8", "d"), c("8", "c"), wc()], HR
        )
        bombs = [r for r in results if r.combination.type == CombinationType.BOMB]
        # Should find a 4-of-a-kind bomb for 8s
        bomb_8 = [b for b in bombs if b.combination.primary_rank == Rank.EIGHT]
        assert len(bomb_8) == 1
        assert bomb_8[0].combination.bomb_size == 4


class TestWildcardStraight:
    def test_wildcard_completes_straight(self):
        # 3,4,5,7 + wildcard → can make 3-4-5-6-7
        results = classify_with_wildcards(
            [c("3", "h"), c("4", "d"), c("5", "c"), c("7", "s"), wc()], HR
        )
        straights = [r for r in results if r.combination.type == CombinationType.STRAIGHT]
        assert len(straights) >= 1
        # Should have a straight with primary rank 7
        s7 = [s for s in straights if s.combination.primary_rank == Rank.SEVEN]
        assert len(s7) == 1


class TestWildcardStraightFlush:
    def test_wildcard_completes_straight_flush(self):
        # 3♠,4♠,5♠,7♠ + wildcard → can make 3-4-5-6-7 all spades
        results = classify_with_wildcards(
            [c("3", "s"), c("4", "s"), c("5", "s"), c("7", "s"), wc()], HR
        )
        sfs = [r for r in results if r.combination.type == CombinationType.STRAIGHT_FLUSH]
        assert len(sfs) >= 1


class TestWildcardFullHouse:
    def test_wildcard_completes_full_house(self):
        # KKK + 3 + wildcard → KKK33
        results = classify_with_wildcards(
            [c("K", "h"), c("K", "d"), c("K", "c"), c("3", "s"), wc()], HR
        )
        fhs = [r for r in results
               if r.combination.type == CombinationType.TRIPLE_PLUS_PAIR]
        # Should find full house with K triple
        k_fh = [f for f in fhs if f.combination.primary_rank == Rank.KING]
        assert len(k_fh) >= 1


class TestWildcardConsecutivePairs:
    def test_wildcard_completes_consecutive_pairs(self):
        # 3,3,4,4,5 + wildcard → 33 44 55
        results = classify_with_wildcards(
            [c("3", "h"), c("3", "d"), c("4", "h"), c("4", "d"), c("5", "s"), wc()], HR
        )
        cps = [r for r in results
               if r.combination.type == CombinationType.CONSECUTIVE_PAIRS]
        assert len(cps) >= 1


class TestWildcardConsecutiveTriples:
    def test_wildcard_completes_consecutive_triples(self):
        # 333 + 44 + wildcard → 333 444
        results = classify_with_wildcards(
            [c("3", "h"), c("3", "d"), c("3", "c"),
             c("4", "h"), c("4", "d"), wc()], HR
        )
        cts = [r for r in results
               if r.combination.type == CombinationType.CONSECUTIVE_TRIPLES]
        assert len(cts) >= 1


class TestMultipleWildcards:
    def test_two_wildcards_complete_bomb(self):
        # With high card = 2, two 2♥ cards are wildcards (double deck)
        # 8♥,8♦ + 2 wildcards → can make 8888
        results = classify_with_wildcards(
            [c("8", "h"), c("8", "d"), wc(), wc()], HR
        )
        bombs = [r for r in results
                 if r.combination.type == CombinationType.BOMB
                 and r.combination.primary_rank == Rank.EIGHT]
        assert len(bombs) == 1


class TestDeduplication:
    def test_no_duplicate_combinations(self):
        """Multiple wildcard assignments producing the same (type, rank, bomb_size)
        should be deduplicated."""
        results = classify_with_wildcards([c("5", "s"), wc()], HR)
        pair_5s = [r for r in results
                   if r.combination.type == CombinationType.PAIR
                   and r.combination.primary_rank == Rank.FIVE]
        assert len(pair_5s) == 1
