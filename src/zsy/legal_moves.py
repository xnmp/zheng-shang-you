"""Legal move generator for Zheng Shang You.

Given a player's hand, their high-card rank, and the current trick
(combination to beat), enumerate all legal plays.

Rules:
- If no active trick (leading), any valid combination is legal
- If active trick exists, must play same type+size that beats it, or a bomb, or pass
- Pass is always legal

Enumerates combinations directly by type for performance.
Wildcards are handled by separating them from the hand and trying
to fill gaps in each combination type.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

from zsy.cards import Card, Rank, Suit, STANDARD_SUITS, STANDARD_RANKS
from zsy.combinations import (
    Combination,
    CombinationType,
    STRAIGHT_RANKS,
)
from zsy.ranking import is_wildcard
from zsy.wildcard import WildcardAssignment


def legal_moves(
    hand: list[Card],
    high_card_rank: Rank,
    active_combo: Combination | None = None,
) -> list[WildcardAssignment]:
    """Enumerate all legal moves from the given hand.

    Returns list of WildcardAssignment objects. Always includes PASS.
    """
    results: list[WildcardAssignment] = [
        WildcardAssignment(Combination.make_pass(), ())
    ]

    if not hand:
        return results

    ctx = _HandContext(hand, high_card_rank)

    if active_combo is None or active_combo.type == CombinationType.PASS:
        results.extend(_all_combinations(ctx))
    else:
        results.extend(_beating_combinations(ctx, active_combo))

    return results


class _HandContext:
    """Pre-analyzed hand for efficient combination enumeration."""
    def __init__(self, hand: list[Card], high_card_rank: Rank) -> None:
        self.hand = hand
        self.high_card_rank = high_card_rank
        self.wildcards = [c for c in hand if is_wildcard(c, high_card_rank)]
        self.non_wilds = [c for c in hand if not is_wildcard(c, high_card_rank)]
        self.n_wild = len(self.wildcards)

        # Group non-wild, non-joker cards by rank
        self.groups: dict[Rank, list[Card]] = {}
        for c in self.non_wilds:
            if not c.is_joker:
                self.groups.setdefault(c.rank, []).append(c)

        self.jokers = [c for c in self.non_wilds if c.is_joker]


def _make_wa(combo: Combination, wc_assignments: tuple[tuple[Card, Card], ...] = ()) -> WildcardAssignment:
    return WildcardAssignment(combo, wc_assignments)


# --- Enumeration by type (wildcard-aware) ---

def _enum_singles(ctx: _HandContext) -> list[WildcardAssignment]:
    results: list[WildcardAssignment] = []
    for rank, cards in ctx.groups.items():
        for card in cards:
            combo = Combination(CombinationType.SINGLE, (card,), rank)
            results.append(_make_wa(combo))
    for j in ctx.jokers:
        combo = Combination(CombinationType.SINGLE, (j,), Rank.JOKER)
        results.append(_make_wa(combo))
    # Wildcards as singles (played at their own supreme rank)
    for wc in ctx.wildcards:
        combo = Combination(CombinationType.SINGLE, (wc,), wc.rank)
        results.append(_make_wa(combo))
    return results


def _enum_pairs(ctx: _HandContext) -> list[WildcardAssignment]:
    results: list[WildcardAssignment] = []

    # Natural pairs
    for rank, cards in ctx.groups.items():
        if len(cards) >= 2:
            for pair in combinations(cards, 2):
                combo = Combination(CombinationType.PAIR, pair, rank)
                results.append(_make_wa(combo))

    # Wildcard completing pairs (wildcard + 1 card of any rank)
    if ctx.n_wild >= 1:
        seen_ranks: set[Rank] = set()
        for rank, cards in ctx.groups.items():
            if rank not in seen_ranks:
                seen_ranks.add(rank)
                # Use first card + first wildcard
                card = cards[0]
                wc = ctx.wildcards[0]
                actual_cards = (card, wc)
                sub_card = Card(rank, _pick_unused_suit(cards))
                combo = Combination(CombinationType.PAIR, (card, sub_card), rank)
                results.append(_make_wa(combo, ((wc, sub_card),)))

    # Wildcard pairs (2 wildcards as a pair of supreme rank)
    if ctx.n_wild >= 2:
        combo = Combination(CombinationType.PAIR, tuple(ctx.wildcards[:2]), ctx.wildcards[0].rank)
        results.append(_make_wa(combo))

    # Joker pairs (same type only)
    small_j = [j for j in ctx.jokers if j.suit == Suit.SMALL_JOKER]
    big_j = [j for j in ctx.jokers if j.suit == Suit.BIG_JOKER]
    for same in (small_j, big_j):
        if len(same) >= 2:
            combo = Combination(CombinationType.PAIR, tuple(same[:2]), Rank.JOKER)
            results.append(_make_wa(combo))

    return results


def _pick_unused_suit(cards: list[Card]) -> Suit:
    """Pick a suit not used by any of the given cards."""
    used = {c.suit for c in cards}
    for s in STANDARD_SUITS:
        if s not in used:
            return s
    return STANDARD_SUITS[0]


def _enum_triples(ctx: _HandContext) -> list[WildcardAssignment]:
    results: list[WildcardAssignment] = []

    for rank, cards in ctx.groups.items():
        # Natural triples
        if len(cards) >= 3:
            for triple in combinations(cards, 3):
                combo = Combination(CombinationType.TRIPLE, triple, rank)
                results.append(_make_wa(combo))

        # Wildcard-assisted triples
        needed = 3 - len(cards)
        if 0 < needed <= ctx.n_wild:
            natural = tuple(cards)
            wcs = ctx.wildcards[:needed]
            subs = _make_substitutes(rank, cards, needed)
            all_cards = natural + tuple(subs)
            combo = Combination(CombinationType.TRIPLE, all_cards, rank)
            assignments = tuple(zip(wcs, subs))
            results.append(_make_wa(combo, assignments))

    return results


def _make_substitutes(rank: Rank, existing: list[Card], count: int) -> list[Card]:
    """Make substitute cards for wildcards: same rank, picking from available suits.

    In double-deck games, suits may repeat. We cycle through unused suits first,
    then reuse suits if needed.
    """
    used = {c.suit for c in existing}
    # Prefer unused suits, then cycle through all suits
    preferred = [s for s in STANDARD_SUITS if s not in used]
    all_suits = preferred + [s for s in STANDARD_SUITS if s in used]
    return [Card(rank, all_suits[i % len(all_suits)]) for i in range(count)]


def _enum_full_houses(ctx: _HandContext) -> list[WildcardAssignment]:
    results: list[WildcardAssignment] = []

    # Only natural full houses for now (wildcard FH is complex, rare in practice)
    for t_rank, t_cards in ctx.groups.items():
        if len(t_cards) >= 3:
            for triple in combinations(t_cards, 3):
                for p_rank, p_cards in ctx.groups.items():
                    if p_rank == t_rank:
                        remaining = [c for c in p_cards if c not in triple]
                        if len(remaining) >= 2:
                            for pair in combinations(remaining, 2):
                                combo = Combination(CombinationType.TRIPLE_PLUS_PAIR, triple + pair, t_rank)
                                results.append(_make_wa(combo))
                    else:
                        if len(p_cards) >= 2:
                            for pair in combinations(p_cards, 2):
                                combo = Combination(CombinationType.TRIPLE_PLUS_PAIR, triple + pair, t_rank)
                                results.append(_make_wa(combo))

    # Wildcard-assisted: wildcard fills missing in triple or pair
    if ctx.n_wild >= 1:
        for t_rank, t_cards in ctx.groups.items():
            t_have = len(t_cards)
            t_need = max(0, 3 - t_have)

            for p_rank, p_cards in ctx.groups.items():
                if p_rank == t_rank:
                    continue
                p_have = len(p_cards)
                p_need = max(0, 2 - p_have)
                total_need = t_need + p_need

                if total_need > 0 and total_need <= ctx.n_wild and t_have + t_need >= 3 and p_have + p_need >= 2:
                    t_natural = tuple(t_cards[:min(t_have, 3)])
                    p_natural = tuple(p_cards[:min(p_have, 2)])
                    t_subs = _make_substitutes(t_rank, t_cards, t_need)
                    p_subs = _make_substitutes(p_rank, p_cards, p_need)
                    all_cards = t_natural + tuple(t_subs) + p_natural + tuple(p_subs)
                    wcs = ctx.wildcards[:total_need]
                    subs = t_subs + p_subs
                    combo = Combination(CombinationType.TRIPLE_PLUS_PAIR, all_cards, t_rank)
                    results.append(_make_wa(combo, tuple(zip(wcs, subs))))

    return results


def _enum_straights(ctx: _HandContext) -> list[WildcardAssignment]:
    results: list[WildcardAssignment] = []

    for start_idx in range(len(STRAIGHT_RANKS) - 4):
        target_ranks = [STRAIGHT_RANKS[start_idx + i] for i in range(5)]
        missing_ranks = [r for r in target_ranks if r not in ctx.groups or len(ctx.groups[r]) == 0]

        if len(missing_ranks) <= ctx.n_wild:
            # Can fill gaps with wildcards
            if len(missing_ranks) == 0:
                # Natural straight — pick one card per rank
                rank_cards = [ctx.groups[r] for r in target_ranks]
                _product_combos(results, CombinationType.STRAIGHT, target_ranks, rank_cards)
            else:
                # Wildcard-assisted
                natural_cards = []
                for r in target_ranks:
                    if r not in missing_ranks:
                        natural_cards.append(ctx.groups[r][0])
                wcs = ctx.wildcards[:len(missing_ranks)]
                subs = [Card(r, Suit.CLUBS) for r in missing_ranks]
                all_cards = tuple(natural_cards + subs)
                combo = Combination(CombinationType.STRAIGHT, all_cards, max(target_ranks))
                results.append(_make_wa(combo, tuple(zip(wcs, subs))))

    return results


def _enum_straight_flushes(ctx: _HandContext) -> list[WildcardAssignment]:
    results: list[WildcardAssignment] = []

    for start_idx in range(len(STRAIGHT_RANKS) - 4):
        target_ranks = [STRAIGHT_RANKS[start_idx + i] for i in range(5)]
        for suit in STANDARD_SUITS:
            missing = []
            present = []
            for r in target_ranks:
                suited = [c for c in ctx.groups.get(r, []) if c.suit == suit]
                if suited:
                    present.append(suited[0])
                else:
                    missing.append(r)

            if len(missing) <= ctx.n_wild:
                if len(missing) == 0:
                    combo = Combination(CombinationType.STRAIGHT_FLUSH, tuple(present), max(target_ranks))
                    results.append(_make_wa(combo))
                else:
                    wcs = ctx.wildcards[:len(missing)]
                    subs = [Card(r, suit) for r in missing]
                    all_cards = tuple(present + subs)
                    combo = Combination(CombinationType.STRAIGHT_FLUSH, all_cards, max(target_ranks))
                    results.append(_make_wa(combo, tuple(zip(wcs, subs))))

    return results


def _product_combos(
    results: list[WildcardAssignment],
    combo_type: CombinationType,
    ranks: list[Rank],
    cards_per_rank: list[list[Card]],
) -> None:
    """Generate all picks of one card per rank position (Cartesian product)."""
    picks: list[list[Card]] = [[]]
    for rank_cards in cards_per_rank:
        new_picks = []
        for existing in picks:
            for card in rank_cards:
                new_picks.append(existing + [card])
        picks = new_picks

    for pick in picks:
        combo = Combination(combo_type, tuple(pick), max(ranks))
        results.append(_make_wa(combo))


def _enum_consecutive_pairs(ctx: _HandContext) -> list[WildcardAssignment]:
    results: list[WildcardAssignment] = []

    for start_idx in range(len(STRAIGHT_RANKS) - 2):
        target_ranks = [STRAIGHT_RANKS[start_idx + i] for i in range(3)]

        # Count how many wildcards needed
        total_need = 0
        feasible = True
        for r in target_ranks:
            have = len(ctx.groups.get(r, []))
            need = max(0, 2 - have)
            if have > 2:
                need = 0
            total_need += need

        if total_need <= ctx.n_wild:
            if total_need == 0:
                # Natural consecutive pairs
                rank_cards = [ctx.groups[r] for r in target_ranks]
                if all(len(rc) >= 2 for rc in rank_cards):
                    pair_options = [list(combinations(rc, 2)) for rc in rank_cards]
                    for p0 in pair_options[0]:
                        for p1 in pair_options[1]:
                            for p2 in pair_options[2]:
                                cards = p0 + p1 + p2
                                combo = Combination(CombinationType.CONSECUTIVE_PAIRS, cards, max(target_ranks))
                                results.append(_make_wa(combo))
            elif total_need > 0:
                # Wildcard-assisted
                natural_cards: list[Card] = []
                subs: list[Card] = []
                for r in target_ranks:
                    have = ctx.groups.get(r, [])
                    natural_cards.extend(have[:2])
                    need = max(0, 2 - len(have))
                    if need > 0:
                        new_subs = _make_substitutes(r, have, need)
                        subs.extend(new_subs)

                wcs = ctx.wildcards[:total_need]
                all_cards = tuple(natural_cards + subs)
                combo = Combination(CombinationType.CONSECUTIVE_PAIRS, all_cards, max(target_ranks))
                results.append(_make_wa(combo, tuple(zip(wcs, subs))))

    return results


def _enum_consecutive_triples(ctx: _HandContext) -> list[WildcardAssignment]:
    results: list[WildcardAssignment] = []

    for start_idx in range(len(STRAIGHT_RANKS) - 1):
        target_ranks = [STRAIGHT_RANKS[start_idx], STRAIGHT_RANKS[start_idx + 1]]

        total_need = 0
        for r in target_ranks:
            have = len(ctx.groups.get(r, []))
            need = max(0, 3 - have)
            total_need += need

        if total_need <= ctx.n_wild:
            if total_need == 0:
                rank_cards = [ctx.groups[r] for r in target_ranks]
                if all(len(rc) >= 3 for rc in rank_cards):
                    triple_options = [list(combinations(rc, 3)) for rc in rank_cards]
                    for t0 in triple_options[0]:
                        for t1 in triple_options[1]:
                            cards = t0 + t1
                            combo = Combination(CombinationType.CONSECUTIVE_TRIPLES, cards, max(target_ranks))
                            results.append(_make_wa(combo))
            elif total_need > 0:
                natural_cards: list[Card] = []
                subs: list[Card] = []
                for r in target_ranks:
                    have = ctx.groups.get(r, [])
                    natural_cards.extend(have[:3])
                    need = max(0, 3 - len(have))
                    if need > 0:
                        new_subs = _make_substitutes(r, have, need)
                        subs.extend(new_subs)

                wcs = ctx.wildcards[:total_need]
                all_cards = tuple(natural_cards + subs)
                combo = Combination(CombinationType.CONSECUTIVE_TRIPLES, all_cards, max(target_ranks))
                results.append(_make_wa(combo, tuple(zip(wcs, subs))))

    return results


def _enum_bombs(ctx: _HandContext) -> list[WildcardAssignment]:
    results: list[WildcardAssignment] = []

    for rank, cards in ctx.groups.items():
        # Natural bombs
        if len(cards) >= 4:
            for size in range(4, len(cards) + 1):
                for bomb in combinations(cards, size):
                    combo = Combination(CombinationType.BOMB, bomb, rank, bomb_size=size)
                    results.append(_make_wa(combo))

        # Wildcard-assisted bombs
        max_bomb = min(len(cards) + ctx.n_wild, 8)
        for target_size in range(max(4, len(cards) + 1), max_bomb + 1):
            needed = target_size - len(cards)
            if 0 < needed <= ctx.n_wild:
                natural = tuple(cards)
                wcs = ctx.wildcards[:needed]
                subs = _make_substitutes(rank, cards, needed)
                all_cards = natural + tuple(subs)
                combo = Combination(CombinationType.BOMB, all_cards, rank, bomb_size=target_size)
                results.append(_make_wa(combo, tuple(zip(wcs, subs))))

    return results


# --- Main entry points ---

def _all_combinations(ctx: _HandContext) -> list[WildcardAssignment]:
    """Generate all valid combinations from the hand (for leading)."""
    results: list[WildcardAssignment] = []
    results.extend(_enum_singles(ctx))
    results.extend(_enum_pairs(ctx))
    results.extend(_enum_triples(ctx))
    results.extend(_enum_full_houses(ctx))
    results.extend(_enum_straights(ctx))
    results.extend(_enum_straight_flushes(ctx))
    results.extend(_enum_consecutive_pairs(ctx))
    results.extend(_enum_consecutive_triples(ctx))
    results.extend(_enum_bombs(ctx))
    return results


def _beating_combinations(
    ctx: _HandContext,
    active: Combination,
) -> list[WildcardAssignment]:
    """Generate all combinations that beat the active combination."""
    results: list[WildcardAssignment] = []

    type_generators = {
        CombinationType.SINGLE: lambda: _enum_singles(ctx),
        CombinationType.PAIR: lambda: _enum_pairs(ctx),
        CombinationType.TRIPLE: lambda: _enum_triples(ctx),
        CombinationType.TRIPLE_PLUS_PAIR: lambda: _enum_full_houses(ctx),
        CombinationType.STRAIGHT: lambda: _enum_straights(ctx),
        CombinationType.STRAIGHT_FLUSH: lambda: _enum_straight_flushes(ctx),
        CombinationType.CONSECUTIVE_PAIRS: lambda: _enum_consecutive_pairs(ctx),
        CombinationType.CONSECUTIVE_TRIPLES: lambda: _enum_consecutive_triples(ctx),
        CombinationType.BOMB: lambda: _enum_bombs(ctx),
    }

    # Same type that beats it
    gen = type_generators.get(active.type)
    if gen is not None:
        for wa in gen():
            if wa.combination.beats(active):
                results.append(wa)

    # Bombs beat non-bombs
    if not active.is_bomb:
        for wa in _enum_bombs(ctx):
            results.append(wa)
        for wa in _enum_straight_flushes(ctx):
            results.append(wa)
    elif active.type == CombinationType.BOMB:
        # Straight flushes that beat this bomb
        for wa in _enum_straight_flushes(ctx):
            if wa.combination.beats(active):
                results.append(wa)
        # Larger bombs (already included via same-type generator above,
        # but those only included same bomb_size; we need all sizes)
        for wa in _enum_bombs(ctx):
            if wa.combination.beats(active):
                # Avoid duplicates with same-type generator
                if wa not in results:
                    results.append(wa)

    return results
