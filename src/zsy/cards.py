"""Card and deck representation for Zheng Shang You."""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from functools import total_ordering
from typing import Sequence


class Suit(IntEnum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3
    SMALL_JOKER = 4
    BIG_JOKER = 5


SUIT_SYMBOLS = {
    Suit.HEARTS: "♥",
    Suit.DIAMONDS: "♦",
    Suit.CLUBS: "♣",
    Suit.SPADES: "♠",
    Suit.SMALL_JOKER: "SJ",
    Suit.BIG_JOKER: "BJ",
}

STANDARD_SUITS = (Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES)


class Rank(IntEnum):
    """Card ranks ordered by game value (3 lowest, 2 highest non-joker)."""
    THREE = 0
    FOUR = 1
    FIVE = 2
    SIX = 3
    SEVEN = 4
    EIGHT = 5
    NINE = 6
    TEN = 7
    JACK = 8
    QUEEN = 9
    KING = 10
    ACE = 11
    TWO = 12
    JOKER = 13


RANK_SYMBOLS = {
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "10",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
    Rank.TWO: "2",
    Rank.JOKER: "★",
}

STANDARD_RANKS = tuple(r for r in Rank if r != Rank.JOKER)


@total_ordering
@dataclass(frozen=True)
class Card:
    """An immutable playing card."""
    rank: Rank
    suit: Suit

    @staticmethod
    def small_joker() -> Card:
        return Card(Rank.JOKER, Suit.SMALL_JOKER)

    @staticmethod
    def big_joker() -> Card:
        return Card(Rank.JOKER, Suit.BIG_JOKER)

    @property
    def is_joker(self) -> bool:
        return self.rank == Rank.JOKER

    def __repr__(self) -> str:
        if self.suit == Suit.SMALL_JOKER:
            return "SmallJoker"
        if self.suit == Suit.BIG_JOKER:
            return "BigJoker"
        return f"{RANK_SYMBOLS[self.rank]}{SUIT_SYMBOLS[self.suit]}"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.suit < other.suit

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))


def _build_single_deck() -> list[Card]:
    """Build one 54-card deck (52 standard + 2 jokers)."""
    cards: list[Card] = []
    for rank in STANDARD_RANKS:
        for suit in STANDARD_SUITS:
            cards.append(Card(rank, suit))
    cards.append(Card.small_joker())
    cards.append(Card.big_joker())
    return cards


class Deck:
    """A shuffled deck of cards (1 or 2 standard 54-card decks)."""

    def __init__(self, num_decks: int = 1) -> None:
        self.cards: list[Card] = []
        for _ in range(num_decks):
            self.cards.extend(_build_single_deck())
        random.shuffle(self.cards)

    def deal(self, num_players: int) -> list[list[Card]]:
        """Deal all cards as evenly as possible to num_players players.

        Players with lower indices receive extra cards when the deck
        doesn't divide evenly.
        """
        hands: list[list[Card]] = [[] for _ in range(num_players)]
        for i, card in enumerate(self.cards):
            hands[i % num_players].append(card)
        return hands
