"""Built-in agents for Zheng Shang You."""
from __future__ import annotations

import random

from zsy.cards import Card
from zsy.combinations import CombinationType
from zsy.game import Agent, GameState
from zsy.jing_gong import JingGongAgent
from zsy.player import Player
from zsy.wildcard import WildcardAssignment


class RandomAgent:
    """Agent that plays a random legal non-pass move when possible.

    Implements both the game Agent and JingGongAgent protocols.
    """

    def choose_move(
        self,
        player: Player,
        game_state: GameState,
        moves: list[WildcardAssignment],
    ) -> WildcardAssignment:
        non_pass = [m for m in moves if m.combination.type != CombinationType.PASS]
        if non_pass:
            return random.choice(non_pass)
        return moves[0]  # pass

    def choose_return_card(
        self,
        player: Player,
        received_card: Card,
        returnable: list[Card],
    ) -> Card:
        return random.choice(returnable)
