"""Jing Gong (进贡) card exchange for Zheng Shang You.

After a round, the loser gives their best card to the winner.
The winner returns any card of their choice that is below 10.

This module implements the exchange logic and agent interface for
choosing which card to return.
"""
from __future__ import annotations

from typing import Protocol

from zsy.cards import Card, Rank
from zsy.player import Player
from zsy.ranking import effective_rank


# Ranks below 10 that the winner can return
RETURNABLE_RANKS = frozenset({
    Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
    Rank.SEVEN, Rank.EIGHT, Rank.NINE,
})


class JingGongAgent(Protocol):
    """Interface for choosing which card to return in Jing Gong."""
    def choose_return_card(
        self,
        player: Player,
        received_card: Card,
        returnable: list[Card],
    ) -> Card: ...


def find_best_card(player: Player) -> Card:
    """Find the best (highest effective rank) card in a player's hand."""
    return max(player.hand, key=lambda c: (effective_rank(c, player.high_card_rank), c.suit))


def find_returnable_cards(player: Player) -> list[Card]:
    """Find all cards below 10 that the winner can return."""
    return [c for c in player.hand if c.rank in RETURNABLE_RANKS]


def execute_jing_gong(
    winner: Player,
    loser: Player,
    return_agent: JingGongAgent,
) -> tuple[Card, Card]:
    """Execute the Jing Gong exchange.

    Args:
        winner: The player who won the round
        loser: The player who lost (finished last)
        return_agent: Agent that decides which card the winner returns

    Returns:
        (tribute_card, return_card) — the card given and the card returned

    Raises:
        ValueError: If exchange cannot be completed (e.g., no returnable cards)
    """
    # Loser gives their best card
    tribute = find_best_card(loser)
    loser.remove_cards([tribute])
    winner.hand.append(tribute)
    winner.hand.sort()

    # Winner returns a card below 10
    returnable = find_returnable_cards(winner)
    if not returnable:
        # Edge case: winner has no cards below 10.
        # In this case, exchange still happens but winner returns their lowest card.
        returnable = sorted(winner.hand, key=lambda c: effective_rank(c, winner.high_card_rank))
        return_card = returnable[0]
    else:
        return_card = return_agent.choose_return_card(winner, tribute, returnable)

    winner.remove_cards([return_card])
    loser.hand.append(return_card)
    loser.hand.sort()

    return tribute, return_card
