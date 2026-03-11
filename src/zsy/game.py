"""Game loop and trick resolution for Zheng Shang You.

A Game manages the state of a single round:
- Players take turns playing combinations or passing
- A trick ends when all other players pass consecutively
- The trick winner leads the next trick
- The game ends when a player empties their hand
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol

from zsy.cards import Card, Deck, Rank
from zsy.combinations import Combination, CombinationType
from zsy.legal_moves import legal_moves
from zsy.player import Player
from zsy.wildcard import WildcardAssignment


class GamePhase(Enum):
    DEALING = auto()
    PLAYING = auto()
    FINISHED = auto()


class Agent(Protocol):
    """Interface for a player agent (human, AI, or random)."""
    def choose_move(
        self,
        player: Player,
        game_state: GameState,
        moves: list[WildcardAssignment],
    ) -> WildcardAssignment: ...


@dataclass(frozen=True)
class PlayRecord:
    """Record of a single play in the game."""
    player_id: int
    combination: Combination
    cards_played: tuple[Card, ...]


@dataclass
class GameState:
    """Observable game state (what a player can see)."""
    current_player: int
    active_combo: Combination | None
    trick_leader: int
    consecutive_passes: int
    cards_remaining: list[int]  # per player
    teams: list[int] | None = None  # team ID per player, or None
    play_history: list[PlayRecord] = field(default_factory=list)


class Game:
    """Manages a single round of Zheng Shang You."""

    def __init__(
        self,
        num_players: int,
        agents: list[Agent],
        num_decks: int | None = None,
        teams: list[int] | None = None,
    ) -> None:
        """
        Args:
            num_players: Number of players
            agents: Agent per player
            num_decks: 1 for ≤3 players, 2 for 4+. Auto-selected if None.
            teams: Optional team assignment per player (e.g., [0, 1, 0, 1]).
                   If one teammate finishes first, the whole team wins.
        """
        if num_decks is None:
            num_decks = 1 if num_players <= 3 else 2

        self.num_players = num_players
        self.agents = agents
        self.players = [Player(id=i) for i in range(num_players)]
        self.num_decks = num_decks
        self.teams = teams
        self.phase = GamePhase.DEALING

        if teams is not None:
            for i, team_id in enumerate(teams):
                self.players[i].team = team_id

        self.current_player = 0
        self.active_combo: Combination | None = None
        self.trick_leader = 0
        self.consecutive_passes = 0
        self.play_history: list[PlayRecord] = []
        self.winner: int | None = None
        self.winning_team: int | None = None
        self.finish_order: list[int] = []

    def deal(self) -> None:
        """Deal cards to all players."""
        deck = Deck(self.num_decks)
        hands = deck.deal(self.num_players)
        for i, hand in enumerate(hands):
            self.players[i].hand = sorted(hand)
        self.phase = GamePhase.PLAYING

    def get_state(self) -> GameState:
        """Get the current observable game state."""
        return GameState(
            current_player=self.current_player,
            active_combo=self.active_combo,
            trick_leader=self.trick_leader,
            consecutive_passes=self.consecutive_passes,
            cards_remaining=[p.card_count for p in self.players],
            teams=self.teams,
            play_history=list(self.play_history),
        )

    def get_legal_moves(self, player_idx: int) -> list[WildcardAssignment]:
        """Get all legal moves for a player."""
        player = self.players[player_idx]
        return legal_moves(
            player.hand,
            player.high_card_rank,
            self.active_combo,
        )

    def play_turn(self) -> PlayRecord | None:
        """Execute one turn: current player chooses and plays a move.

        Returns the PlayRecord, or None if game is finished.
        """
        if self.phase != GamePhase.PLAYING:
            return None

        player = self.players[self.current_player]

        # Skip players who have already finished
        if not player.has_cards:
            self._advance_player()
            return self.play_turn()

        moves = self.get_legal_moves(self.current_player)
        agent = self.agents[self.current_player]
        chosen = agent.choose_move(player, self.get_state(), moves)

        return self._apply_move(chosen)

    def _apply_move(self, wa: WildcardAssignment) -> PlayRecord:
        """Apply a chosen move to the game state."""
        player = self.players[self.current_player]
        combo = wa.combination

        # Determine actual cards played (from the player's hand)
        if combo.type == CombinationType.PASS:
            actual_cards: tuple[Card, ...] = ()
        else:
            # The cards in wa.combination.cards may be substituted versions.
            # We need to remove the actual hand cards (non-wildcards + wildcards).
            actual_cards = self._extract_actual_cards(wa, player)
            player.remove_cards(list(actual_cards))

        record = PlayRecord(
            player_id=self.current_player,
            combination=combo,
            cards_played=actual_cards,
        )
        self.play_history.append(record)

        if combo.type == CombinationType.PASS:
            self.consecutive_passes += 1
        else:
            self.active_combo = combo
            self.trick_leader = self.current_player
            self.consecutive_passes = 0

        # Check if player won
        if not player.has_cards and player.id not in self.finish_order:
            self.finish_order.append(player.id)
            if self.winner is None:
                self.winner = player.id
                if self.teams is not None:
                    self.winning_team = self.teams[player.id]

        # Check if trick is over (all other active players passed)
        active_players = [p for p in self.players if p.has_cards]
        # N-1 passes means everyone except the trick leader passed
        others_count = len(active_players) - (1 if self.players[self.trick_leader].has_cards else 0)
        if self.consecutive_passes >= others_count:
            # Trick is over, trick leader leads next
            self.active_combo = None
            self.consecutive_passes = 0
            self.current_player = self.trick_leader
            # If trick leader has no cards, find next active player
            if not self.players[self.trick_leader].has_cards:
                self._advance_player()
        else:
            self._advance_player()

        # Check if game is over (only 1 or 0 players with cards)
        if len(active_players) <= 1:
            # Add remaining players to finish order
            for p in self.players:
                if p.id not in self.finish_order:
                    self.finish_order.append(p.id)
            self.phase = GamePhase.FINISHED

        return record

    def _extract_actual_cards(
        self, wa: WildcardAssignment, player: Player
    ) -> tuple[Card, ...]:
        """Determine which actual hand cards are used in this play.

        For wildcard assignments, we need the original wildcard cards,
        not their substituted identities.
        """
        if not wa.assignments:
            return wa.combination.cards

        # Build list: non-wildcard cards + original wildcard cards
        substituted = {sub_card for _, sub_card in wa.assignments}
        wildcard_originals = [orig for orig, _ in wa.assignments]
        non_wild = [c for c in wa.combination.cards if c not in substituted]

        return tuple(non_wild + wildcard_originals)

    def _advance_player(self) -> None:
        """Move to the next player who still has cards."""
        for _ in range(self.num_players):
            self.current_player = (self.current_player + 1) % self.num_players
            if self.players[self.current_player].has_cards:
                return
        # No active players left — should not reach here normally

    def run(self) -> int:
        """Run the game to completion. Returns winner's player ID."""
        if self.phase == GamePhase.DEALING:
            self.deal()

        while self.phase == GamePhase.PLAYING:
            self.play_turn()

        assert self.winner is not None
        return self.winner
