"""Card combination types and validation for Zheng Shang You.

Legal combinations:
- Single: 1 card
- Pair: 2 cards of same rank
- Triple: 3 cards of same rank
- TriplePlusPair (full house): 3+2 of different ranks
- Straight: exactly 5 consecutive ranks (no 2s or jokers in sequence)
- ConsecutivePairs: exactly 3 consecutive pairs (6 cards)
- ConsecutiveTriples: exactly 2 consecutive triples (6 cards)
- Bomb: 4+ cards of same rank
- StraightFlush: exactly 5 consecutive cards of the same suit
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

from zsy.cards import Card, Rank, Suit, STANDARD_SUITS


class CombinationType(Enum):
    SINGLE = auto()
    PAIR = auto()
    TRIPLE = auto()
    TRIPLE_PLUS_PAIR = auto()
    STRAIGHT = auto()
    CONSECUTIVE_PAIRS = auto()
    CONSECUTIVE_TRIPLES = auto()
    BOMB = auto()
    STRAIGHT_FLUSH = auto()
    PASS = auto()


# Ranks that can appear in straights (3 through A, no 2, no joker)
STRAIGHT_RANKS = tuple(r for r in Rank if r not in (Rank.TWO, Rank.JOKER))


def _are_consecutive(ranks: list[Rank]) -> bool:
    """Check if sorted ranks form a consecutive sequence within STRAIGHT_RANKS."""
    if not ranks:
        return False
    sorted_ranks = sorted(ranks)
    for r in sorted_ranks:
        if r not in STRAIGHT_RANKS:
            return False
    for i in range(1, len(sorted_ranks)):
        if sorted_ranks[i].value - sorted_ranks[i - 1].value != 1:
            return False
    return True


@dataclass(frozen=True)
class Combination:
    """A validated card combination."""
    type: CombinationType
    cards: tuple[Card, ...]
    # The primary rank used for comparison (e.g., the triple's rank in a full house)
    primary_rank: Rank | None
    # For bombs: the count of cards (4, 5, 6, etc.)
    bomb_size: int = 0

    @staticmethod
    def make_pass() -> Combination:
        return Combination(type=CombinationType.PASS, cards=(), primary_rank=None)

    @property
    def is_bomb(self) -> bool:
        return self.type in (CombinationType.BOMB, CombinationType.STRAIGHT_FLUSH)

    def beats(self, other: Combination) -> bool:
        """Check if this combination can beat the other.

        Rules:
        - PASS never beats anything
        - Bombs beat all non-bombs
        - Bomb hierarchy: higher count > lower count, except straight flush
          beats 5-of-a-kind but NOT 6-of-a-kind
        - Same type: must have same size, higher primary rank wins
        """
        if self.type == CombinationType.PASS:
            return False
        if other.type == CombinationType.PASS:
            return True

        # Bomb vs non-bomb
        if self.is_bomb and not other.is_bomb:
            return True
        if not self.is_bomb and other.is_bomb:
            return False

        # Both bombs
        if self.is_bomb and other.is_bomb:
            return _bomb_beats(self, other)

        # Same type, same card count required
        if self.type != other.type:
            return False
        if len(self.cards) != len(other.cards):
            return False

        assert self.primary_rank is not None and other.primary_rank is not None
        return self.primary_rank > other.primary_rank


def _bomb_beats(attacker: Combination, defender: Combination) -> bool:
    """Compare two bombs.

    Hierarchy:
    - 6+ of a kind > straight flush > 5 of a kind > 4 of a kind
    - Within same category, higher rank wins
    - Within bomb of same count, higher rank wins
    """
    a_power = _bomb_power(attacker)
    d_power = _bomb_power(defender)
    if a_power != d_power:
        return a_power > d_power
    # Same power tier: compare by primary rank
    assert attacker.primary_rank is not None and defender.primary_rank is not None
    return attacker.primary_rank > defender.primary_rank


def _bomb_power(bomb: Combination) -> int:
    """Assign a power tier to a bomb for ordering.

    4-of-a-kind = 1
    5-of-a-kind = 2
    straight flush = 3  (beats 5-of-a-kind but not 6-of-a-kind)
    6-of-a-kind = 4
    7-of-a-kind = 5
    8-of-a-kind = 6
    ...
    """
    if bomb.type == CombinationType.STRAIGHT_FLUSH:
        return 3
    # BOMB type
    if bomb.bomb_size <= 5:
        return bomb.bomb_size - 3  # 4->1, 5->2
    return bomb.bomb_size - 2  # 6->4, 7->5, 8->6


def classify(cards: Sequence[Card]) -> Combination | None:
    """Classify a set of cards as a valid combination, or None if invalid.

    Does not account for wildcards — those are handled at a higher level.
    """
    n = len(cards)
    card_tuple = tuple(cards)

    if n == 0:
        return Combination.make_pass()

    rank_counts = Counter(c.rank for c in cards)
    suit_counts = Counter(c.suit for c in cards)

    if n == 1:
        return Combination(CombinationType.SINGLE, card_tuple, cards[0].rank)

    if n == 2:
        if len(rank_counts) == 1 and not cards[0].is_joker:
            return Combination(CombinationType.PAIR, card_tuple, cards[0].rank)
        # Joker pair: both must be same type (both small or both big)
        if cards[0].is_joker and cards[1].is_joker:
            if cards[0].suit == cards[1].suit:
                return Combination(CombinationType.PAIR, card_tuple, Rank.JOKER)
        return None

    if n == 3:
        if len(rank_counts) == 1 and not cards[0].is_joker:
            return Combination(CombinationType.TRIPLE, card_tuple, cards[0].rank)
        return None

    # 4+ of a kind → bomb
    if len(rank_counts) == 1 and not cards[0].is_joker:
        if n >= 4:
            return Combination(CombinationType.BOMB, card_tuple, cards[0].rank, bomb_size=n)
        return None

    if n == 5:
        # Check straight flush first
        sf = _check_straight_flush(cards)
        if sf is not None:
            return sf

        # Check straight
        st = _check_straight(cards)
        if st is not None:
            return st

        # Check triple + pair (full house)
        fh = _check_full_house(cards)
        if fh is not None:
            return fh

        return None

    if n == 6:
        # Check consecutive pairs (3 consecutive pairs)
        cp = _check_consecutive_pairs(cards)
        if cp is not None:
            return cp

        # Check consecutive triples (2 consecutive triples)
        ct = _check_consecutive_triples(cards)
        if ct is not None:
            return ct

        return None

    return None


def _check_straight(cards: Sequence[Card]) -> Combination | None:
    """Check if exactly 5 cards form a straight (consecutive ranks, no 2/joker)."""
    if len(cards) != 5:
        return None
    ranks = [c.rank for c in cards]
    if any(c.is_joker for c in cards):
        return None
    if len(set(ranks)) != 5:
        return None
    if _are_consecutive(ranks):
        return Combination(CombinationType.STRAIGHT, tuple(cards), max(ranks))
    return None


def _check_straight_flush(cards: Sequence[Card]) -> Combination | None:
    """Check if exactly 5 cards form a straight flush."""
    if len(cards) != 5:
        return None
    if any(c.is_joker for c in cards):
        return None
    suits = {c.suit for c in cards}
    if len(suits) != 1:
        return None
    ranks = [c.rank for c in cards]
    if len(set(ranks)) != 5:
        return None
    if _are_consecutive(ranks):
        return Combination(CombinationType.STRAIGHT_FLUSH, tuple(cards), max(ranks))
    return None


def _check_full_house(cards: Sequence[Card]) -> Combination | None:
    """Check if 5 cards form triple + pair."""
    if len(cards) != 5:
        return None
    rank_counts = Counter(c.rank for c in cards)
    if sorted(rank_counts.values()) != [2, 3]:
        return None
    if any(c.is_joker for c in cards):
        return None
    triple_rank = next(r for r, cnt in rank_counts.items() if cnt == 3)
    return Combination(CombinationType.TRIPLE_PLUS_PAIR, tuple(cards), triple_rank)


def _check_consecutive_pairs(cards: Sequence[Card]) -> Combination | None:
    """Check if 6 cards form 3 consecutive pairs."""
    if len(cards) != 6:
        return None
    if any(c.is_joker for c in cards):
        return None
    rank_counts = Counter(c.rank for c in cards)
    if len(rank_counts) != 3 or not all(v == 2 for v in rank_counts.values()):
        return None
    ranks = sorted(rank_counts.keys())
    if _are_consecutive(ranks):
        return Combination(CombinationType.CONSECUTIVE_PAIRS, tuple(cards), max(ranks))
    return None


def _check_consecutive_triples(cards: Sequence[Card]) -> Combination | None:
    """Check if 6 cards form 2 consecutive triples."""
    if len(cards) != 6:
        return None
    if any(c.is_joker for c in cards):
        return None
    rank_counts = Counter(c.rank for c in cards)
    if len(rank_counts) != 2 or not all(v == 3 for v in rank_counts.values()):
        return None
    ranks = sorted(rank_counts.keys())
    if _are_consecutive(ranks):
        return Combination(CombinationType.CONSECUTIVE_TRIPLES, tuple(cards), max(ranks))
    return None
