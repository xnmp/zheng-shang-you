"""State and action encoding for RL training (DouZero-style).

Encodes game states and actions as flat numerical vectors suitable for
neural network input. Uses card-matrix representation where each card
position is indexed by (rank, suit_index).

Dimensions:
- Card matrix: NUM_RANKS × NUM_SUITS = 15 × 4 = 60 values (counts 0-8)
  - 13 standard ranks (3 through 2) + 2 joker slots
  - 4 suits for standard cards; joker slots use index 0 only
- State vector components (for 3-player, single-deck):
  - Current hand: 60
  - Other players' played cards: (N-1) × 60
  - Last K moves: K × (60 + N)  (card matrix + one-hot player ID)
  - Cards remaining per player: N
  - High-card rank per player: N × NUM_STANDARD_RANKS (one-hot)
  - Wildcard in hand: 1
  - Bombs played this game: 1
  - Team flags (optional): N
"""
from __future__ import annotations

import numpy as np

from zsy.cards import Card, Rank, Suit, STANDARD_RANKS, STANDARD_SUITS
from zsy.combinations import Combination, CombinationType
from zsy.game import GameState, PlayRecord
from zsy.player import Player
from zsy.ranking import is_wildcard
from zsy.wildcard import WildcardAssignment


# --- Constants ---

NUM_STANDARD_RANKS = 13  # 3, 4, ..., K, A, 2
NUM_RANKS = 15  # 13 standard + small joker + big joker
NUM_SUITS = 4  # Hearts, Diamonds, Clubs, Spades
CARD_MATRIX_SIZE = NUM_RANKS * NUM_SUITS  # 60

# Joker encoding: row 13 = small joker, row 14 = big joker, column 0
SMALL_JOKER_IDX = (13, 0)
BIG_JOKER_IDX = (14, 0)

MOVE_HISTORY_K = 15  # Number of recent moves to encode


def card_index(card: Card) -> tuple[int, int]:
    """Map a Card to (rank_idx, suit_idx) in the card matrix.

    rank_idx: 0-12 for standard ranks (3..2), 13 for small joker, 14 for big joker
    suit_idx: 0-3 for standard suits (Hearts, Diamonds, Clubs, Spades)
    """
    if card.suit == Suit.SMALL_JOKER:
        return SMALL_JOKER_IDX
    if card.suit == Suit.BIG_JOKER:
        return BIG_JOKER_IDX
    return (card.rank.value, int(card.suit))


def cards_to_matrix(cards: list[Card] | tuple[Card, ...]) -> np.ndarray:
    """Encode a collection of cards as a NUM_RANKS × NUM_SUITS count matrix.

    Each cell contains the count of that specific (rank, suit) card.
    For single-deck games, values are 0 or 1.
    For double-deck games, values can be 0, 1, or 2.
    """
    matrix = np.zeros((NUM_RANKS, NUM_SUITS), dtype=np.float32)
    for card in cards:
        r, s = card_index(card)
        matrix[r, s] += 1.0
    return matrix


def cards_to_vector(cards: list[Card] | tuple[Card, ...]) -> np.ndarray:
    """Encode cards as a flat vector of length CARD_MATRIX_SIZE."""
    return cards_to_matrix(cards).flatten()


# --- Action Encoding ---

def encode_action(wa: WildcardAssignment) -> np.ndarray:
    """Encode an action (WildcardAssignment) as a flat card-matrix vector.

    Pass is encoded as the zero vector.
    For non-pass moves, the combination's cards are placed in the matrix.
    Wildcard substitutions are encoded at their substituted position
    (the position they're pretending to be).
    """
    combo = wa.combination
    if combo.type == CombinationType.PASS:
        return np.zeros(CARD_MATRIX_SIZE, dtype=np.float32)
    return cards_to_vector(combo.cards)


# --- State Encoding ---

def encode_state(
    player: Player,
    game_state: GameState,
    num_players: int,
    all_played_cards: list[list[Card]] | None = None,
) -> np.ndarray:
    """Encode the full game state from a player's perspective.

    Args:
        player: The current player (with hand and high_card_rank)
        game_state: Observable game state
        num_players: Total number of players
        all_played_cards: Per-player list of cards they've played so far.
            If None, reconstructed from play_history.

    Returns:
        Flat numpy vector representing the state.
    """
    parts: list[np.ndarray] = []

    # 1. Current hand (60D)
    parts.append(cards_to_vector(player.hand))

    # 2. Played cards per OTHER player (N-1 × 60D)
    played = _get_played_cards(game_state, num_players, all_played_cards)
    player_id = player.id
    for i in range(num_players):
        if i != player_id:
            parts.append(cards_to_vector(played[i]))

    # 3. Last K moves (K × (60 + num_players))
    history = game_state.play_history
    for k in range(MOVE_HISTORY_K):
        idx = len(history) - 1 - k
        if idx >= 0:
            record = history[idx]
            move_vec = cards_to_vector(record.combination.cards)
            player_onehot = np.zeros(num_players, dtype=np.float32)
            player_onehot[record.player_id] = 1.0
            parts.append(np.concatenate([move_vec, player_onehot]))
        else:
            parts.append(np.zeros(CARD_MATRIX_SIZE + num_players, dtype=np.float32))

    # 4. Cards remaining per player (N)
    remaining = np.array(game_state.cards_remaining, dtype=np.float32)
    # Normalize by initial hand size (54 / num_players for single deck)
    parts.append(remaining)

    # 5. High-card rank per player (N × NUM_STANDARD_RANKS one-hot)
    # We only know our own high-card rank in imperfect info, but
    # in ZSY the high-card rank is public knowledge (advances on wins)
    # For now, encode current player's rank; others default to Rank.TWO
    for i in range(num_players):
        rank_onehot = np.zeros(NUM_STANDARD_RANKS, dtype=np.float32)
        if i == player_id:
            rank_onehot[player.high_card_rank.value] = 1.0
        else:
            rank_onehot[Rank.TWO.value] = 1.0  # default
        parts.append(rank_onehot)

    # 6. Wildcard in hand (1)
    has_wildcard = any(is_wildcard(c, player.high_card_rank) for c in player.hand)
    parts.append(np.array([1.0 if has_wildcard else 0.0], dtype=np.float32))

    # 7. Bomb count this game (1)
    bomb_count = sum(
        1 for r in game_state.play_history
        if r.combination.is_bomb
    )
    parts.append(np.array([float(bomb_count)], dtype=np.float32))

    # 8. Team flags (N) - optional
    if game_state.teams is not None:
        my_team = game_state.teams[player_id]
        team_flags = np.array(
            [1.0 if game_state.teams[i] == my_team else -1.0 for i in range(num_players)],
            dtype=np.float32,
        )
        parts.append(team_flags)

    return np.concatenate(parts)


def state_dim(num_players: int, teams: bool = False) -> int:
    """Calculate the state vector dimensionality for a given player count."""
    dim = CARD_MATRIX_SIZE  # hand
    dim += (num_players - 1) * CARD_MATRIX_SIZE  # others' played cards
    dim += MOVE_HISTORY_K * (CARD_MATRIX_SIZE + num_players)  # move history
    dim += num_players  # cards remaining
    dim += num_players * NUM_STANDARD_RANKS  # high-card ranks
    dim += 1  # wildcard flag
    dim += 1  # bomb count
    if teams:
        dim += num_players  # team flags
    return dim


def action_dim() -> int:
    """Action vector dimensionality (fixed regardless of player count)."""
    return CARD_MATRIX_SIZE


def _get_played_cards(
    game_state: GameState,
    num_players: int,
    all_played_cards: list[list[Card]] | None,
) -> list[list[Card]]:
    """Get per-player played cards, reconstructing from history if needed."""
    if all_played_cards is not None:
        return all_played_cards

    played: list[list[Card]] = [[] for _ in range(num_players)]
    for record in game_state.play_history:
        played[record.player_id].extend(record.cards_played)
    return played
