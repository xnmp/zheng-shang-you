"""Wildcard mechanics for Zheng Shang You.

The heart-suit card of a player's high-card rank is a wildcard that can
substitute for any card in any combination, including bombs and straight
flushes.

This module provides:
- classify_with_wildcards: classify a set of cards that may include wildcards
- A WildcardAssignment that records how wildcards are used
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from zsy.cards import Card, Rank, Suit, STANDARD_SUITS, STANDARD_RANKS
from zsy.combinations import (
    Combination,
    CombinationType,
    STRAIGHT_RANKS,
    _are_consecutive,
    classify,
)
from zsy.ranking import is_wildcard


@dataclass(frozen=True)
class WildcardAssignment:
    """Records how wildcards are used in a combination."""
    combination: Combination
    # Maps wildcard card -> the card it's substituting for
    assignments: tuple[tuple[Card, Card], ...]


def classify_with_wildcards(
    cards: Sequence[Card],
    high_card_rank: Rank,
) -> list[WildcardAssignment]:
    """Classify cards that may include wildcards into all valid combinations.

    Returns a list of WildcardAssignment objects, one per valid interpretation.
    If no wildcards are present, returns at most one result (the normal classification).
    """
    card_list = list(cards)
    wildcards = [c for c in card_list if is_wildcard(c, high_card_rank)]
    non_wildcards = [c for c in card_list if not is_wildcard(c, high_card_rank)]
    n_wild = len(wildcards)

    if n_wild == 0:
        combo = classify(card_list)
        if combo is not None:
            return [WildcardAssignment(combo, ())]
        return []

    # With wildcards, try substituting them to form valid combinations
    results: list[WildcardAssignment] = []
    n = len(card_list)

    if n == 0:
        return [WildcardAssignment(Combination.make_pass(), ())]

    # Generate candidate substitutions based on combination type and size
    seen: set[tuple[CombinationType, Rank | None, int]] = set()

    for sub_cards in _wildcard_substitutions(non_wildcards, n_wild, n):
        combo = classify(sub_cards)
        if combo is not None:
            # Deduplicate by (type, primary_rank, bomb_size) to avoid
            # strategically equivalent combinations
            key = (combo.type, combo.primary_rank, combo.bomb_size)
            if key not in seen:
                seen.add(key)
                # Build assignment mapping
                assignments = tuple(
                    (wildcards[i], sub_cards[len(non_wildcards) + i])
                    for i in range(n_wild)
                )
                results.append(WildcardAssignment(combo, assignments))

    return results


def _wildcard_substitutions(
    non_wildcards: list[Card],
    n_wild: int,
    total: int,
) -> list[list[Card]]:
    """Generate candidate card lists by substituting wildcards with real cards.

    Rather than trying all 54^n_wild possibilities, we generate candidates
    intelligently based on what could form valid combinations.
    """
    candidates: list[list[Card]] = []

    if total == 1:
        # Wildcard as any single card — but it's just a single, so use it as-is
        # Actually, as a single it plays at its own rank (supreme rank)
        # For strategic purposes, it's usually played as itself
        for rank in STANDARD_RANKS:
            for suit in STANDARD_SUITS:
                candidates.append(non_wildcards + [Card(rank, suit)])
        return candidates

    # For each possible combination size, generate plausible substitutions
    if total == 2:
        # Could form a pair: match existing card's rank, or pick any rank
        _add_rank_fill_candidates(candidates, non_wildcards, n_wild, count_needed=2)

    elif total == 3:
        _add_rank_fill_candidates(candidates, non_wildcards, n_wild, count_needed=3)

    elif total == 4:
        # Could be a bomb
        _add_rank_fill_candidates(candidates, non_wildcards, n_wild, count_needed=4)

    elif total == 5:
        # Could be straight, straight flush, full house, or 5-bomb
        _add_rank_fill_candidates(candidates, non_wildcards, n_wild, count_needed=5)
        _add_straight_candidates(candidates, non_wildcards, n_wild)
        _add_straight_flush_candidates(candidates, non_wildcards, n_wild)
        _add_full_house_candidates(candidates, non_wildcards, n_wild)

    elif total == 6:
        # Could be consecutive pairs, consecutive triples, or 6-bomb
        _add_rank_fill_candidates(candidates, non_wildcards, n_wild, count_needed=6)
        _add_consecutive_pair_candidates(candidates, non_wildcards, n_wild)
        _add_consecutive_triple_candidates(candidates, non_wildcards, n_wild)

    elif total >= 4:
        # Large bombs
        _add_rank_fill_candidates(candidates, non_wildcards, n_wild, count_needed=total)

    return candidates


def _add_rank_fill_candidates(
    out: list[list[Card]],
    non_wilds: list[Card],
    n_wild: int,
    count_needed: int,
) -> None:
    """Add candidates where wildcards fill out N-of-a-kind."""
    rank_counts = Counter(c.rank for c in non_wilds if not c.is_joker)

    for rank in STANDARD_RANKS:
        existing = rank_counts.get(rank, 0)
        needed = count_needed - existing
        if 0 < needed <= n_wild:
            # Fill with cards of this rank using different suits
            used_suits = {c.suit for c in non_wilds if c.rank == rank}
            available_suits = [s for s in STANDARD_SUITS if s not in used_suits]
            if len(available_suits) >= needed:
                fill = [Card(rank, s) for s in available_suits[:needed]]
                out.append(non_wilds + fill)


def _add_straight_candidates(
    out: list[list[Card]],
    non_wilds: list[Card],
    n_wild: int,
) -> None:
    """Add candidates where wildcards complete a 5-card straight."""
    existing_ranks = {c.rank for c in non_wilds if c.rank in STRAIGHT_RANKS}

    for start_idx in range(len(STRAIGHT_RANKS) - 4):
        target_ranks = [STRAIGHT_RANKS[start_idx + i] for i in range(5)]
        missing = [r for r in target_ranks if r not in existing_ranks]
        if len(missing) == n_wild:
            fill = [Card(r, Suit.CLUBS) for r in missing]  # Suit doesn't matter for straight
            out.append(non_wilds + fill)


def _add_straight_flush_candidates(
    out: list[list[Card]],
    non_wilds: list[Card],
    n_wild: int,
) -> None:
    """Add candidates where wildcards complete a 5-card straight flush."""
    if not non_wilds:
        # All wildcards — try every possible straight flush
        for suit in STANDARD_SUITS:
            for start_idx in range(len(STRAIGHT_RANKS) - 4):
                target_ranks = [STRAIGHT_RANKS[start_idx + i] for i in range(5)]
                out.append([Card(r, suit) for r in target_ranks])
        return

    suits = {c.suit for c in non_wilds}
    existing_ranks = {c.rank for c in non_wilds}

    for suit in suits:
        suit_ranks = {c.rank for c in non_wilds if c.suit == suit}
        non_suit = [c for c in non_wilds if c.suit != suit]

        for start_idx in range(len(STRAIGHT_RANKS) - 4):
            target_ranks = [STRAIGHT_RANKS[start_idx + i] for i in range(5)]
            # Cards in this suit that match the target
            matching = suit_ranks & set(target_ranks)
            # Cards not in this suit need to be covered by wildcards too
            needed = 5 - len(matching) - len(non_suit)
            # Actually: we need all 5 cards of this suit. Non-wild cards of
            # wrong suit can't be part of a straight flush.
            suit_cards_in_target = [c for c in non_wilds if c.suit == suit and c.rank in target_ranks]
            other_cards = [c for c in non_wilds if c not in suit_cards_in_target]
            # Other cards can't contribute to this straight flush
            if other_cards:
                continue
            missing_count = 5 - len(suit_cards_in_target)
            if missing_count == n_wild:
                missing_ranks = [r for r in target_ranks if r not in suit_ranks]
                fill = [Card(r, suit) for r in missing_ranks]
                out.append(suit_cards_in_target + fill)


def _add_full_house_candidates(
    out: list[list[Card]],
    non_wilds: list[Card],
    n_wild: int,
) -> None:
    """Add candidates where wildcards complete a full house (triple + pair)."""
    rank_counts = Counter(c.rank for c in non_wilds if not c.is_joker)

    for triple_rank in STANDARD_RANKS:
        for pair_rank in STANDARD_RANKS:
            if triple_rank == pair_rank:
                continue
            t_have = rank_counts.get(triple_rank, 0)
            p_have = rank_counts.get(pair_rank, 0)
            t_need = max(0, 3 - t_have)
            p_need = max(0, 2 - p_have)
            if t_need + p_need == n_wild and t_have + t_need == 3 and p_have + p_need == 2:
                used_t_suits = {c.suit for c in non_wilds if c.rank == triple_rank}
                used_p_suits = {c.suit for c in non_wilds if c.rank == pair_rank}
                avail_t = [s for s in STANDARD_SUITS if s not in used_t_suits]
                avail_p = [s for s in STANDARD_SUITS if s not in used_p_suits]
                if len(avail_t) >= t_need and len(avail_p) >= p_need:
                    fill = [Card(triple_rank, s) for s in avail_t[:t_need]]
                    fill += [Card(pair_rank, s) for s in avail_p[:p_need]]
                    out.append(non_wilds + fill)


def _add_consecutive_pair_candidates(
    out: list[list[Card]],
    non_wilds: list[Card],
    n_wild: int,
) -> None:
    """Add candidates where wildcards complete 3 consecutive pairs."""
    rank_counts = Counter(c.rank for c in non_wilds if not c.is_joker)

    for start_idx in range(len(STRAIGHT_RANKS) - 2):
        target_ranks = [STRAIGHT_RANKS[start_idx + i] for i in range(3)]
        total_needed = 0
        feasible = True
        for r in target_ranks:
            have = rank_counts.get(r, 0)
            need = max(0, 2 - have)
            if have > 2:
                feasible = False
                break
            total_needed += need
        if feasible and total_needed == n_wild:
            fill: list[Card] = []
            for r in target_ranks:
                have = rank_counts.get(r, 0)
                need = max(0, 2 - have)
                used_suits = {c.suit for c in non_wilds if c.rank == r}
                avail = [s for s in STANDARD_SUITS if s not in used_suits]
                if len(avail) < need:
                    feasible = False
                    break
                fill.extend(Card(r, s) for s in avail[:need])
            if feasible:
                out.append(non_wilds + fill)


def _add_consecutive_triple_candidates(
    out: list[list[Card]],
    non_wilds: list[Card],
    n_wild: int,
) -> None:
    """Add candidates where wildcards complete 2 consecutive triples."""
    rank_counts = Counter(c.rank for c in non_wilds if not c.is_joker)

    for start_idx in range(len(STRAIGHT_RANKS) - 1):
        target_ranks = [STRAIGHT_RANKS[start_idx], STRAIGHT_RANKS[start_idx + 1]]
        total_needed = 0
        feasible = True
        for r in target_ranks:
            have = rank_counts.get(r, 0)
            need = max(0, 3 - have)
            if have > 3:
                feasible = False
                break
            total_needed += need
        if feasible and total_needed == n_wild:
            fill: list[Card] = []
            for r in target_ranks:
                have = rank_counts.get(r, 0)
                need = max(0, 3 - have)
                used_suits = {c.suit for c in non_wilds if c.rank == r}
                avail = [s for s in STANDARD_SUITS if s not in used_suits]
                if len(avail) < need:
                    feasible = False
                    break
                fill.extend(Card(r, s) for s in avail[:need])
            if feasible:
                out.append(non_wilds + fill)
