"""Player state for Zheng Shang You."""
from __future__ import annotations

from dataclasses import dataclass, field

from zsy.cards import Card, Rank


@dataclass
class Player:
    """Mutable player state within a game."""
    id: int
    hand: list[Card] = field(default_factory=list)
    high_card_rank: Rank = Rank.TWO
    team: int | None = None

    @property
    def card_count(self) -> int:
        return len(self.hand)

    @property
    def has_cards(self) -> bool:
        return len(self.hand) > 0

    def remove_cards(self, cards: list[Card]) -> None:
        """Remove the given cards from hand. Raises ValueError if not present."""
        remaining = list(self.hand)
        for card in cards:
            remaining.remove(card)  # raises ValueError if not found
        self.hand = remaining

    def advance_high_card(self) -> None:
        """Increment the high card rank by 1 (called when this player wins).

        Wraps around after TWO back to THREE.
        """
        if self.high_card_rank == Rank.TWO:
            self.high_card_rank = Rank.THREE
        else:
            self.high_card_rank = Rank(self.high_card_rank.value + 1)
