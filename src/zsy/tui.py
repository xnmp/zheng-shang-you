"""Interactive TUI for playing Zheng Shang You."""
from __future__ import annotations

import random
from collections.abc import Awaitable

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Center
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Label, Static, Log

from zsy.cards import Card, Rank, Suit, RANK_SYMBOLS
from zsy.combinations import CombinationType
from zsy.game import Game, GamePhase, GameState, Agent
from zsy.agents import RandomAgent
from zsy.player import Player
from zsy.ranking import effective_rank, is_wildcard
from zsy.wildcard import WildcardAssignment


# --- Card display ---

SUIT_GLYPHS = {Suit.HEARTS: "♥", Suit.DIAMONDS: "♦", Suit.CLUBS: "♣", Suit.SPADES: "♠"}

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


def rich_card(card: Card, high_card_rank: Rank | None = None) -> str:
    """Return a rich-markup string for a card."""
    if card.suit == Suit.SMALL_JOKER:
        return "[bold magenta]🃏sm[/]"
    if card.suit == Suit.BIG_JOKER:
        return "[bold red]🃏BG[/]"

    rank_s = RANK_SYMBOLS[card.rank]
    suit_s = SUIT_GLYPHS[card.suit]

    if high_card_rank and is_wildcard(card, high_card_rank):
        return f"[bold yellow on dark_red]★{rank_s}{suit_s}[/]"

    if card.suit in (Suit.HEARTS, Suit.DIAMONDS):
        return f"[bold red]{rank_s}{suit_s}[/]"
    return f"[bold white]{rank_s}{suit_s}[/]"


def rich_combo(wa: WildcardAssignment, hr: Rank | None = None) -> str:
    """Rich-markup for a combination."""
    combo = wa.combination
    if combo.type == CombinationType.PASS:
        return "[dim]Pass[/]"
    label = COMBO_LABELS.get(combo.type, "?")
    cards = " ".join(rich_card(c, hr) for c in combo.cards)
    extra = ""
    if combo.is_bomb:
        extra = " [bold red]💣[/]"
    if wa.assignments:
        wc_info = " ".join(f"{repr(o)}→{repr(s)}" for o, s in wa.assignments)
        extra += f" [dim]\\[wc: {wc_info}][/]"
    return f"[bold]{label}[/]{extra}  {cards}"


# --- Widgets ---

class CardWidget(Static):
    """A single card that can be selected."""
    selected = reactive(False)

    def __init__(self, card: Card, index: int, high_card_rank: Rank, **kw) -> None:
        super().__init__(**kw)
        self.card = card
        self.index = index
        self.high_card_rank = high_card_rank

    def render(self) -> str:
        r = RANK_SYMBOLS[self.card.rank] if not self.card.is_joker else ("sm" if self.card.suit == Suit.SMALL_JOKER else "BG")
        s = SUIT_GLYPHS.get(self.card.suit, "🃏")
        return f" {r}{s} "

    def _get_classes(self) -> str:
        cls = "card"
        if self.card.suit in (Suit.HEARTS, Suit.DIAMONDS):
            cls += " red-card"
        elif self.card.is_joker:
            cls += " joker-card"
        if is_wildcard(self.card, self.high_card_rank):
            cls += " wildcard"
        if self.selected:
            cls += " selected"
        return cls

    def watch_selected(self) -> None:
        self.set_classes(self._get_classes())

    def on_mount(self) -> None:
        self.set_classes(self._get_classes())

    def on_click(self) -> None:
        self.selected = not self.selected


class MoveButton(Button):
    """A button representing a legal move."""
    def __init__(self, wa: WildcardAssignment, index: int, hr: Rank, **kw) -> None:
        self.wa = wa
        self.move_index = index
        label = rich_combo(wa, hr)
        variant = "default"
        if wa.combination.type == CombinationType.PASS:
            variant = "default"
        elif wa.combination.is_bomb:
            variant = "error"
        super().__init__(label, variant=variant, **kw)


# --- Main App ---

CSS = """
Screen {
    layout: vertical;
}

#table-area {
    height: auto;
    max-height: 14;
    border: solid green;
    padding: 1 2;
    margin: 0 1;
}

#opponents {
    height: auto;
    padding: 0 2;
    margin: 0 1;
}

.opponent-info {
    margin: 0 2;
}

#hand-area {
    height: auto;
    border: solid dodgerblue;
    padding: 1 1;
    margin: 0 1;
}

#hand-cards {
    height: auto;
    layout: horizontal;
    overflow-x: auto;
}

.card {
    width: 6;
    height: 3;
    content-align: center middle;
    border: round white;
    margin: 0 0;
}

.red-card {
    color: red;
    border: round red;
}

.joker-card {
    color: magenta;
    border: round magenta;
}

.wildcard {
    color: yellow;
    border: heavy yellow;
    background: $surface;
}

.selected {
    background: $accent;
    border: heavy $accent;
}

#moves-area {
    height: auto;
    max-height: 12;
    border: solid $warning;
    padding: 1 1;
    margin: 0 1;
    overflow-y: auto;
}

#moves-list {
    layout: vertical;
    height: auto;
}

#moves-list Button {
    margin: 0 0;
    width: 100%;
    height: 3;
}

#game-log {
    height: 1fr;
    min-height: 6;
    border: solid $surface-lighten-2;
    margin: 0 1;
}
"""


class ZSYApp(App):
    TITLE = "争上游 Zheng Shang You"
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("p", "play_pass", "Pass"),
        Binding("up,k", "move_up", "Prev Move", show=False),
        Binding("down,j", "move_down", "Next Move", show=False),
        Binding("enter", "confirm_move", "Play Move", show=False),
    ]

    def __init__(
        self,
        num_players: int = 3,
        human_player: int = 0,
        num_decks: int | None = None,
        teams: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_players = num_players
        self.human_player = human_player
        self.num_decks = num_decks
        self.teams = teams
        self.game: Game | None = None
        self._pending_moves: list[WildcardAssignment] = []
        self._move_buttons: list[MoveButton] = []
        self._waiting_for_input = False
        self._selected_move: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="table-area")
        yield Horizontal(id="opponents")
        yield Vertical(
            Label("[bold]Your Hand[/]"),
            Horizontal(id="hand-cards"),
            id="hand-area",
        )
        yield Vertical(
            Label("[bold]Legal Moves[/]"),
            Vertical(id="moves-list"),
            id="moves-area",
        )
        yield Log(id="game-log")
        yield Footer()

    def on_mount(self) -> None:
        self._start_game()

    def _start_game(self) -> None:
        agents: list[Agent] = []
        for i in range(self.num_players):
            if i == self.human_player:
                agents.append(self)  # placeholder, we handle turns manually
            else:
                agents.append(RandomAgent())

        self.game = Game(
            self.num_players, agents,
            num_decks=self.num_decks, teams=self.teams,
        )
        self.game.deal()

        log = self.query_one("#game-log", Log)
        log.write_line(f"Game started: {self.num_players} players, {self.game.num_decks} deck(s)")
        if self.teams:
            log.write_line(f"Teams: {self.teams}")
        log.write_line(f"You are Player {self.human_player}")
        log.write_line("")

        self._run_until_human()

    def _update_display(self) -> None:
        if self.game is None:
            return

        game = self.game
        hr = game.players[self.human_player].high_card_rank
        state = game.get_state()

        # Table area — active combo
        table = self.query_one("#table-area", Static)
        if state.active_combo and state.active_combo.type != CombinationType.PASS:
            wa = WildcardAssignment(state.active_combo, ())
            table.update(
                f"[bold]Current Trick[/]  —  Player {state.trick_leader} played:  "
                f"{rich_combo(wa, hr)}"
            )
        else:
            table.update("[bold green]Table is clear — lead with anything![/]")

        # Opponents
        opp_container = self.query_one("#opponents", Horizontal)
        opp_container.remove_children()
        for i in range(self.num_players):
            if i == self.human_player:
                continue
            count = state.cards_remaining[i]
            team_tag = ""
            if self.teams is not None:
                p_team = self.teams[self.human_player]
                if self.teams[i] == p_team:
                    team_tag = " [green](ally)[/]"
                else:
                    team_tag = " [red](enemy)[/]"
            finished = " [bold green]✓ OUT[/]" if count == 0 else ""
            lbl = Label(
                f"P{i}: [bold]{count}[/] cards{team_tag}{finished}",
                classes="opponent-info",
            )
            opp_container.mount(lbl)

        # Hand
        hand_container = self.query_one("#hand-cards", Horizontal)
        hand_container.remove_children()
        player = game.players[self.human_player]
        sorted_hand = sorted(player.hand, key=lambda c: (effective_rank(c, hr), c.suit))
        for i, card in enumerate(sorted_hand):
            w = CardWidget(card, i, hr)
            hand_container.mount(w)

    def _show_moves(self, moves: list[WildcardAssignment]) -> None:
        """Display legal moves as buttons."""
        self._pending_moves = moves
        self._waiting_for_input = True
        hr = self.game.players[self.human_player].high_card_rank if self.game else Rank.TWO

        moves_list = self.query_one("#moves-list", Vertical)
        moves_list.remove_children()
        self._move_buttons = []
        for i, m in enumerate(moves):
            btn = MoveButton(m, i, hr)
            self._move_buttons.append(btn)
            moves_list.mount(btn)
        self._selected_move = 0
        if self._move_buttons:
            self.set_timer(0.05, self._focus_selected_move)

    def _clear_moves(self) -> None:
        moves_list = self.query_one("#moves-list", Vertical)
        moves_list.remove_children()
        self._move_buttons = []
        self._waiting_for_input = False

    @on(Button.Pressed)
    def on_move_pressed(self, event: Button.Pressed) -> None:
        if not isinstance(event.button, MoveButton):
            return
        if not self._waiting_for_input:
            return
        self._play_human_move(event.button.wa)

    def action_play_pass(self) -> None:
        if not self._waiting_for_input or not self._pending_moves:
            return
        pass_move = self._pending_moves[0]  # pass is always first
        if pass_move.combination.type == CombinationType.PASS:
            self._play_human_move(pass_move)

    def _focus_selected_move(self) -> None:
        if 0 <= self._selected_move < len(self._move_buttons):
            self._move_buttons[self._selected_move].focus()

    def action_move_up(self) -> None:
        if not self._waiting_for_input or not self._pending_moves:
            return
        self._selected_move = (self._selected_move - 1) % len(self._pending_moves)
        self._focus_selected_move()

    def action_move_down(self) -> None:
        if not self._waiting_for_input or not self._pending_moves:
            return
        self._selected_move = (self._selected_move + 1) % len(self._pending_moves)
        self._focus_selected_move()

    def action_confirm_move(self) -> None:
        if not self._waiting_for_input or not self._pending_moves:
            return
        if 0 <= self._selected_move < len(self._pending_moves):
            self._play_human_move(self._pending_moves[self._selected_move])

    def _play_human_move(self, wa: WildcardAssignment) -> None:
        if self.game is None:
            return
        self._waiting_for_input = False
        self._clear_moves()

        log = self.query_one("#game-log", Log)
        hr = self.game.players[self.human_player].high_card_rank
        combo = wa.combination

        if combo.type == CombinationType.PASS:
            log.write_line("[You] Pass")
        else:
            cards_str = " ".join(repr(c) for c in combo.cards)
            label = COMBO_LABELS.get(combo.type, "?")
            log.write_line(f"[You] {label}: {cards_str}")

        # Apply the move
        self.game._apply_move(wa)

        player = self.game.players[self.human_player]
        if not player.has_cards:
            log.write_line("*** You are out of cards! ***")

        self._update_display()

        if self.game.phase == GamePhase.FINISHED:
            self._show_game_over()
        else:
            self._run_until_human()

    def _run_until_human(self) -> None:
        """Run AI turns until it's the human's turn."""
        if self.game is None:
            return

        log = self.query_one("#game-log", Log)

        while self.game.phase == GamePhase.PLAYING:
            if self.game.current_player == self.human_player:
                player = self.game.players[self.human_player]
                if not player.has_cards:
                    self.game._advance_player()
                    continue
                # Human's turn
                moves = self.game.get_legal_moves(self.human_player)
                self._update_display()
                self._show_moves(moves)
                return

            # AI turn
            player = self.game.players[self.game.current_player]
            if not player.has_cards:
                self.game._advance_player()
                continue

            pid = self.game.current_player
            moves = self.game.get_legal_moves(pid)
            agent = self.game.agents[pid]
            chosen = agent.choose_move(player, self.game.get_state(), moves)
            combo = chosen.combination

            self.game._apply_move(chosen)

            if combo.type == CombinationType.PASS:
                log.write_line(f"  P{pid} passes")
            else:
                cards_str = " ".join(repr(c) for c in combo.cards)
                label = COMBO_LABELS.get(combo.type, "?")
                bomb = " 💣" if combo.is_bomb else ""
                log.write_line(f"  P{pid} plays {label}: {cards_str}{bomb}")

            if not player.has_cards:
                log.write_line(f"  *** P{pid} is out of cards! ***")

            self._update_display()

        self._show_game_over()

    def _show_game_over(self) -> None:
        if self.game is None:
            return
        log = self.query_one("#game-log", Log)
        log.write_line("")
        log.write_line("=" * 40)
        log.write_line("  GAME OVER")
        log.write_line(f"  Winner: Player {self.game.winner}")
        log.write_line(f"  Finish order: {self.game.finish_order}")
        if self.game.winning_team is not None:
            log.write_line(f"  Winning team: {self.game.winning_team}")
        if self.game.winner == self.human_player:
            log.write_line("  🎉 YOU WIN! 🎉")
        else:
            log.write_line("  You lost.")
        log.write_line("=" * 40)
        log.write_line("Press 'q' to quit")

        table = self.query_one("#table-area", Static)
        if self.game.winner == self.human_player:
            table.update("[bold green]🎉 YOU WIN! 🎉[/]")
        else:
            table.update(f"[bold red]Game Over — Player {self.game.winner} wins[/]")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Play Zheng Shang You (TUI)")
    parser.add_argument("-n", "--players", type=int, default=3, help="Number of players (2-6)")
    parser.add_argument("-p", "--player", type=int, default=0, help="Your player index")
    parser.add_argument("-d", "--decks", type=int, default=None, help="Number of decks")
    parser.add_argument("-t", "--teams", type=int, nargs="*", default=None,
                        help="Team assignments (e.g., 0 1 0 1)")
    args = parser.parse_args()

    app = ZSYApp(
        num_players=args.players,
        human_player=args.player,
        num_decks=args.decks,
        teams=args.teams,
    )
    app.run()


if __name__ == "__main__":
    main()
