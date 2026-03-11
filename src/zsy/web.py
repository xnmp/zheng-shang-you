"""Web GUI server for Zheng Shang You using FastAPI + WebSocket."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from zsy.cards import Card, Rank, Suit, RANK_SYMBOLS, SUIT_SYMBOLS
from zsy.combinations import CombinationType
from zsy.game import Game, GamePhase, GameState, Agent
from zsy.agents import RandomAgent
from zsy.player import Player
from zsy.ranking import effective_rank, is_wildcard
from zsy.wildcard import WildcardAssignment

STATIC_DIR = Path(__file__).parent / "static"

COMBO_LABELS = {
    CombinationType.SINGLE: "Single",
    CombinationType.PAIR: "Pair",
    CombinationType.TRIPLE: "Triple",
    CombinationType.TRIPLE_PLUS_PAIR: "Full House",
    CombinationType.STRAIGHT: "Straight",
    CombinationType.CONSECUTIVE_PAIRS: "Consec. Pairs",
    CombinationType.CONSECUTIVE_TRIPLES: "Consec. Triples",
    CombinationType.BOMB: "BOMB",
    CombinationType.STRAIGHT_FLUSH: "STRAIGHT FLUSH",
    CombinationType.PASS: "Pass",
}


def card_to_dict(card: Card, high_card_rank: Rank) -> dict:
    """Serialize a Card to a JSON-friendly dict."""
    return {
        "rank": card.rank.value,
        "suit": card.suit.value,
        "rank_symbol": RANK_SYMBOLS.get(card.rank, "?"),
        "suit_symbol": SUIT_SYMBOLS.get(card.suit, "?"),
        "is_joker": card.is_joker,
        "is_wildcard": is_wildcard(card, high_card_rank),
        "is_red": card.suit in (Suit.HEARTS, Suit.DIAMONDS),
        "effective_rank": effective_rank(card, high_card_rank),
        "id": f"{card.rank.value}-{card.suit.value}",
    }


def _actual_hand_card_ids(wa: WildcardAssignment) -> list[str]:
    """Compute the card IDs that this move consumes from the player's hand."""
    combo = wa.combination
    if combo.type == CombinationType.PASS:
        return []
    if not wa.assignments:
        return [f"{c.rank.value}-{c.suit.value}" for c in combo.cards]
    # With wildcards: non-substituted cards + original wildcard cards
    substituted = {sub for _, sub in wa.assignments}
    non_wild = [c for c in combo.cards if c not in substituted]
    originals = [orig for orig, _ in wa.assignments]
    return [f"{c.rank.value}-{c.suit.value}" for c in non_wild + originals]


def move_to_dict(wa: WildcardAssignment, index: int, high_card_rank: Rank) -> dict:
    """Serialize a WildcardAssignment to a JSON-friendly dict."""
    combo = wa.combination
    cards = [card_to_dict(c, high_card_rank) for c in combo.cards]
    return {
        "index": index,
        "type": combo.type.name,
        "label": COMBO_LABELS.get(combo.type, "?"),
        "cards": cards,
        "is_bomb": combo.is_bomb,
        "is_pass": combo.type == CombinationType.PASS,
        "hand_card_ids": _actual_hand_card_ids(wa),
        "assignments": [
            {"original": card_to_dict(o, high_card_rank),
             "substitute": card_to_dict(s, high_card_rank)}
            for o, s in wa.assignments
        ],
    }


def game_state_to_dict(
    game: Game, human_player: int
) -> dict:
    """Build the full game state dict to send to the client."""
    state = game.get_state()
    hr = game.players[human_player].high_card_rank
    player = game.players[human_player]

    # Sort hand by effective rank
    sorted_hand = sorted(player.hand, key=lambda c: (effective_rank(c, hr), c.suit.value))

    # Active combo
    active = None
    if state.active_combo and state.active_combo.type != CombinationType.PASS:
        active = {
            "label": COMBO_LABELS.get(state.active_combo.type, "?"),
            "cards": [card_to_dict(c, hr) for c in state.active_combo.cards],
            "is_bomb": state.active_combo.is_bomb,
            "player": state.trick_leader,
        }

    return {
        "type": "state",
        "hand": [card_to_dict(c, hr) for c in sorted_hand],
        "active_combo": active,
        "current_player": state.current_player,
        "trick_leader": state.trick_leader,
        "cards_remaining": state.cards_remaining,
        "human_player": human_player,
        "num_players": game.num_players,
        "phase": game.phase.name,
        "is_human_turn": (
            game.phase == GamePhase.PLAYING
            and state.current_player == human_player
            and player.has_cards
        ),
    }


class GameSession:
    """Manages a single game session with a WebSocket client."""

    def __init__(
        self,
        ws: WebSocket,
        num_players: int = 3,
        human_player: int = 0,
        num_decks: int | None = None,
    ) -> None:
        self.ws = ws
        self.num_players = num_players
        self.human_player = human_player
        self.num_decks = num_decks
        self.game: Game | None = None
        self._move_queue: asyncio.Queue[int] = asyncio.Queue()
        self._game_task: asyncio.Task | None = None

    async def send(self, msg: dict) -> None:
        await self.ws.send_json(msg)

    async def send_log(self, text: str) -> None:
        await self.send({"type": "log", "text": text})

    async def start(self) -> None:
        """Start a new game and run the game loop."""
        agents: list[Agent] = []
        for i in range(self.num_players):
            if i == self.human_player:
                agents.append(RandomAgent())  # placeholder, not used for human
            else:
                agents.append(RandomAgent())

        self.game = Game(
            self.num_players, agents,
            num_decks=self.num_decks,
        )
        self.game.deal()

        await self.send_log(
            f"Game started: {self.num_players} players, "
            f"{self.game.num_decks} deck(s). You are Player {self.human_player}."
        )
        await self._run_game_loop()

    async def _run_game_loop(self) -> None:
        """Run the game, pausing for human input when needed."""
        game = self.game
        assert game is not None

        while game.phase == GamePhase.PLAYING:
            pid = game.current_player
            player = game.players[pid]

            if not player.has_cards:
                game._advance_player()
                continue

            if pid == self.human_player:
                # Human turn — send state + legal moves, wait for choice
                moves = game.get_legal_moves(pid)
                hr = player.high_card_rank

                # Auto-pass if the only move is pass
                non_pass = [m for m in moves if m.combination.type != CombinationType.PASS]
                if not non_pass:
                    chosen = moves[0]
                    await self.send(game_state_to_dict(game, self.human_player))
                    await self.send({"type": "auto_pass"})
                    await self.send_log("[You] Pass (no playable moves)")
                    game._apply_move(chosen)
                    await asyncio.sleep(0.5)
                else:
                    await self.send(game_state_to_dict(game, self.human_player))
                    await self.send({
                        "type": "moves",
                        "moves": [move_to_dict(m, i, hr) for i, m in enumerate(moves)],
                    })

                    # Wait for the human to choose
                    choice_idx = await self._move_queue.get()
                    if 0 <= choice_idx < len(moves):
                        chosen = moves[choice_idx]
                    else:
                        chosen = moves[0]

                    combo = chosen.combination
                    if combo.type == CombinationType.PASS:
                        await self.send_log("[You] Pass")
                    else:
                        cards_str = " ".join(repr(c) for c in combo.cards)
                        label = COMBO_LABELS.get(combo.type, "?")
                        await self.send_log(f"[You] {label}: {cards_str}")

                    game._apply_move(chosen)

                if not player.has_cards:
                    await self.send_log("*** You are out of cards! ***")
                    await self.send(game_state_to_dict(game, self.human_player))
                    await self.send({
                        "type": "human_finished",
                        "winner": game.winner == self.human_player,
                    })

            else:
                # AI turn
                moves = game.get_legal_moves(pid)
                agent = game.agents[pid]
                chosen = agent.choose_move(player, game.get_state(), moves)
                combo = chosen.combination

                game._apply_move(chosen)

                if combo.type == CombinationType.PASS:
                    await self.send_log(f"  P{pid} passes")
                else:
                    cards_str = " ".join(repr(c) for c in combo.cards)
                    label = COMBO_LABELS.get(combo.type, "?")
                    bomb = " \U0001f4a3" if combo.is_bomb else ""
                    await self.send_log(f"  P{pid} plays {label}: {cards_str}{bomb}")

                if not player.has_cards:
                    await self.send_log(f"  *** P{pid} is out of cards! ***")

                # Small delay so AI moves feel natural
                await asyncio.sleep(0.3)

            # Send updated state after each move
            await self.send(game_state_to_dict(game, self.human_player))

        # Game over
        await self._send_game_over()

    async def _send_game_over(self) -> None:
        game = self.game
        assert game is not None

        won = game.winner == self.human_player
        await self.send_log("")
        await self.send_log("=" * 40)
        await self.send_log("  GAME OVER")
        await self.send_log(f"  Winner: Player {game.winner}")
        await self.send_log(f"  Finish order: {game.finish_order}")
        if won:
            await self.send_log("  YOU WIN!")
        else:
            await self.send_log("  You lost.")
        await self.send_log("=" * 40)

        await self.send({
            "type": "game_over",
            "winner": game.winner,
            "human_won": won,
            "finish_order": game.finish_order,
        })

    def start_in_background(self) -> None:
        """Cancel any running game and start a new one as a background task."""
        if self._game_task and not self._game_task.done():
            self._game_task.cancel()
        self._move_queue = asyncio.Queue()
        self._game_task = asyncio.create_task(self._run_start())

    async def _run_start(self) -> None:
        try:
            await self.start()
        except asyncio.CancelledError:
            pass

    async def handle_message(self, data: dict) -> None:
        """Handle an incoming WebSocket message from the client."""
        msg_type = data.get("type")

        if msg_type == "play_move":
            move_index = data.get("index", 0)
            await self._move_queue.put(move_index)

        elif msg_type == "new_game":
            self.start_in_background()


app = FastAPI(title="Zheng Shang You")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = GameSession(websocket, num_players=3, human_player=0)
    session.start_in_background()

    try:
        while True:
            data = await websocket.receive_json()
            await session.handle_message(data)
    except WebSocketDisconnect:
        if session._game_task and not session._game_task.done():
            session._game_task.cancel()


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Play Zheng Shang You (Web GUI)")
    parser.add_argument("-n", "--players", type=int, default=3,
                        help="Number of players (default: 3)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    # Store config for the WebSocket handler
    app.state.num_players = args.players

    print(f"Starting Zheng Shang You Web GUI on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
