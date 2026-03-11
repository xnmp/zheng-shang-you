"""Built-in agents for Zheng Shang You."""
from __future__ import annotations

import random

from zsy.cards import Card
from zsy.combinations import CombinationType
from zsy.game import Agent, GameState
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


class HeuristicAgent:
    """Rule-based heuristic agent that prioritizes shedding cards efficiently.

    Strategy:
    - When leading: play smallest single, pair, or triple to conserve strong cards
    - When following: play the lowest-rank legal combination that beats the trick
    - Save bombs for when they're needed
    - Pass when only bombs are available (unless hand is small)
    """

    def choose_move(
        self,
        player: Player,
        game_state: GameState,
        moves: list[WildcardAssignment],
    ) -> WildcardAssignment:
        non_pass = [m for m in moves if m.combination.type != CombinationType.PASS]
        if not non_pass:
            return moves[0]  # pass

        is_leading = game_state.active_combo is None

        # Separate bombs from non-bombs
        bombs = [m for m in non_pass if m.combination.is_bomb]
        non_bombs = [m for m in non_pass if not m.combination.is_bomb]

        if is_leading:
            return self._lead(player, non_bombs, bombs)
        else:
            return self._follow(player, non_bombs, bombs, moves[0])

    def _lead(
        self,
        player: Player,
        non_bombs: list[WildcardAssignment],
        bombs: list[WildcardAssignment],
    ) -> WildcardAssignment:
        """Choose what to lead with. Prefer small combos, low ranks."""
        if non_bombs:
            # Prefer moves that shed more cards, but with lower rank
            # Sort by: (-card_count, primary_rank) to shed more cards first,
            # then by lowest rank
            return min(
                non_bombs,
                key=lambda m: (
                    -len(m.combination.cards),  # more cards = better
                    m.combination.primary_rank or 0,  # lower rank = better
                ),
            )
        # Only bombs available — play the smallest one
        return min(bombs, key=lambda m: (m.combination.bomb_size, m.combination.primary_rank or 0))

    def _follow(
        self,
        player: Player,
        non_bombs: list[WildcardAssignment],
        bombs: list[WildcardAssignment],
        pass_move: WildcardAssignment,
    ) -> WildcardAssignment:
        """Choose the minimum-rank legal play to beat the trick."""
        if non_bombs:
            # Play the lowest-rank card that beats the trick
            return min(
                non_bombs,
                key=lambda m: m.combination.primary_rank or 0,
            )
        # Only bombs beat — use them only if hand is small (≤6 cards)
        if bombs and player.card_count <= 6:
            return min(bombs, key=lambda m: (m.combination.bomb_size, m.combination.primary_rank or 0))
        return pass_move

    def choose_return_card(
        self,
        player: Player,
        received_card: Card,
        returnable: list[Card],
    ) -> Card:
        # Return the lowest-rank returnable card
        return min(returnable, key=lambda c: c.rank)
