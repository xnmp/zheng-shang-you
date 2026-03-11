"""Dynamic card ranking system for Zheng Shang You.

Each player has a "high card" rank (supreme rank) that starts at 2 and
advances when they win. Cards of that rank outrank all other standard cards.
The heart-suit card of the high-card rank is additionally a wildcard.

This module provides functions for:
- Determining effective rank ordering given a player's high-card rank
- Identifying wildcard cards
- Comparing cards under a specific player's ranking rules
"""
from __future__ import annotations

from zsy.cards import Card, Rank, Suit, STANDARD_RANKS


def effective_rank(card: Card, high_card_rank: Rank) -> int:
    """Return a comparable integer for a card's effective rank.

    Ordering (low to high):
    - Standard ranks excluding the high card, in normal order
    - The high card rank (supreme — above all standard ranks)
    - Small joker
    - Big joker

    The high card of hearts (wildcard) has the same effective rank as
    other suits of the high card — its special power is substitution,
    not raw rank.
    """
    if card.suit == Suit.BIG_JOKER:
        return 100
    if card.suit == Suit.SMALL_JOKER:
        return 99

    if card.rank == high_card_rank:
        return 98  # Supreme rank, above all other standard cards

    return card.rank.value


def is_wildcard(card: Card, high_card_rank: Rank) -> bool:
    """Check if a card is the heart-suit wildcard for the given high-card rank."""
    return card.rank == high_card_rank and card.suit == Suit.HEARTS


def get_wildcards(cards: list[Card], high_card_rank: Rank) -> list[Card]:
    """Return all wildcard cards from a hand."""
    return [c for c in cards if is_wildcard(c, high_card_rank)]


def is_high_card(card: Card, high_card_rank: Rank) -> bool:
    """Check if a card is of the supreme rank (any suit)."""
    return card.rank == high_card_rank and not card.is_joker


def compare_cards(a: Card, b: Card, high_card_rank: Rank) -> int:
    """Compare two cards under a player's ranking rules.

    Returns negative if a < b, zero if equal rank, positive if a > b.
    """
    ea = effective_rank(a, high_card_rank)
    eb = effective_rank(b, high_card_rank)
    return ea - eb
