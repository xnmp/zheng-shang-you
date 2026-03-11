"""Legal move generator for Zheng Shang You.

Given a player's hand, their high-card rank, and the current trick
(combination to beat), enumerate all legal plays.

Rules:
- If no active trick (leading), any valid combination is legal
- If active trick exists, must play same type+size that beats it, or a bomb, or pass
- Pass is always legal
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

from zsy.cards import Card, Rank, Suit, STANDARD_SUITS, STANDARD_RANKS
from zsy.combinations import (
    Combination,
    CombinationType,
    STRAIGHT_RANKS,
    classify,
)
from zsy.ranking import is_wildcard
from zsy.wildcard import classify_with_wildcards, WildcardAssignment


def legal_moves(
    hand: list[Card],
    high_card_rank: Rank,
    active_combo: Combination | None = None,
) -> list[WildcardAssignment]:
    """Enumerate all legal moves from the given hand.

    Args:
        hand: Cards in the player's hand
        high_card_rank: This player's current high-card rank
        active_combo: The combination to beat, or None if leading

    Returns:
        List of WildcardAssignment objects representing legal plays.
        Always includes PASS.
    """
    results: list[WildcardAssignment] = []
    pass_wa = WildcardAssignment(Combination.make_pass(), ())
    results.append(pass_wa)

    if not hand:
        return results

    if active_combo is None or active_combo.type == CombinationType.PASS:
        # Leading: any valid combination is legal
        results.extend(_all_combinations(hand, high_card_rank))
    else:
        # Must beat the active combination
        results.extend(_beating_combinations(hand, high_card_rank, active_combo))

    return results


def _all_combinations(
    hand: list[Card],
    high_card_rank: Rank,
) -> list[WildcardAssignment]:
    """Generate all valid combinations from the hand (for leading)."""
    results: list[WildcardAssignment] = []
    seen: set[tuple[CombinationType, Rank | None, int, tuple[Card, ...]]] = set()

    for size in range(1, min(len(hand), 8) + 1):
        for card_subset in combinations(hand, size):
            for wa in classify_with_wildcards(list(card_subset), high_card_rank):
                combo = wa.combination
                if combo.type == CombinationType.PASS:
                    continue
                key = (combo.type, combo.primary_rank, combo.bomb_size, tuple(sorted(card_subset)))
                if key not in seen:
                    seen.add(key)
                    results.append(wa)

    return results


def _beating_combinations(
    hand: list[Card],
    high_card_rank: Rank,
    active: Combination,
) -> list[WildcardAssignment]:
    """Generate all combinations that beat the active combination."""
    results: list[WildcardAssignment] = []
    seen: set[tuple[CombinationType, Rank | None, int, tuple[Card, ...]]] = set()
    target_size = len(active.cards)

    # Try same-size combinations that beat it
    if target_size <= len(hand):
        for card_subset in combinations(hand, target_size):
            for wa in classify_with_wildcards(list(card_subset), high_card_rank):
                combo = wa.combination
                if combo.type == CombinationType.PASS:
                    continue
                if combo.beats(active):
                    key = (combo.type, combo.primary_rank, combo.bomb_size, tuple(sorted(card_subset)))
                    if key not in seen:
                        seen.add(key)
                        results.append(wa)

    # If active is not a bomb, also try all bomb sizes
    if not active.is_bomb:
        for size in range(4, min(len(hand), 8) + 1):
            if size == target_size:
                continue  # Already covered above
            for card_subset in combinations(hand, size):
                for wa in classify_with_wildcards(list(card_subset), high_card_rank):
                    combo = wa.combination
                    if combo.is_bomb:
                        key = (combo.type, combo.primary_rank, combo.bomb_size, tuple(sorted(card_subset)))
                        if key not in seen:
                            seen.add(key)
                            results.append(wa)

    # If active is a bomb, try larger bombs
    if active.is_bomb:
        for size in range(target_size + 1, min(len(hand), 8) + 1):
            for card_subset in combinations(hand, size):
                for wa in classify_with_wildcards(list(card_subset), high_card_rank):
                    combo = wa.combination
                    if combo.is_bomb and combo.beats(active):
                        key = (combo.type, combo.primary_rank, combo.bomb_size, tuple(sorted(card_subset)))
                        if key not in seen:
                            seen.add(key)
                            results.append(wa)
        # Also try straight flush (5 cards) beating bombs
        if target_size != 5:
            for card_subset in combinations(hand, 5):
                for wa in classify_with_wildcards(list(card_subset), high_card_rank):
                    combo = wa.combination
                    if combo.type == CombinationType.STRAIGHT_FLUSH and combo.beats(active):
                        key = (combo.type, combo.primary_rank, combo.bomb_size, tuple(sorted(card_subset)))
                        if key not in seen:
                            seen.add(key)
                            results.append(wa)

    return results
