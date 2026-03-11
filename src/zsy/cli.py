"""Interactive CLI for playing Zheng Shang You against AI opponents."""
from __future__ import annotations

import sys
from zsy.cards import Card, Rank, Suit, RANK_SYMBOLS, SUIT_SYMBOLS
from zsy.combinations import CombinationType
from zsy.game import Game, GamePhase, GameState, Agent
from zsy.agents import RandomAgent
from zsy.player import Player
from zsy.ranking import is_wildcard, effective_rank
from zsy.wildcard import WildcardAssignment


# --- Display helpers ---

COMBO_NAMES = {
    CombinationType.SINGLE: "Single",
    CombinationType.PAIR: "Pair",
    CombinationType.TRIPLE: "Triple",
    CombinationType.TRIPLE_PLUS_PAIR: "Full House",
    CombinationType.STRAIGHT: "Straight",
    CombinationType.CONSECUTIVE_PAIRS: "Consec. Pairs",
    CombinationType.CONSECUTIVE_TRIPLES: "Consec. Triples",
    CombinationType.BOMB: "Bomb",
    CombinationType.STRAIGHT_FLUSH: "Straight Flush",
    CombinationType.PASS: "Pass",
}


def card_str(card: Card, high_card_rank: Rank | None = None) -> str:
    """Colorized card string."""
    base = repr(card)
    if high_card_rank and is_wildcard(card, high_card_rank):
        return f"\033[1;33m{base}★\033[0m"  # yellow bold for wildcard
    if card.suit in (Suit.HEARTS, Suit.DIAMONDS):
        return f"\033[91m{base}\033[0m"  # red
    if card.is_joker:
        return f"\033[1;35m{base}\033[0m"  # magenta
    return base


def hand_str(hand: list[Card], high_card_rank: Rank) -> str:
    """Display hand with indices."""
    sorted_hand = sorted(hand, key=lambda c: (effective_rank(c, high_card_rank), c.suit))
    parts = []
    for i, c in enumerate(sorted_hand):
        parts.append(f"  {i:2d}: {card_str(c, high_card_rank)}")
    return "\n".join(parts)


def combo_str(wa: WildcardAssignment, high_card_rank: Rank | None = None) -> str:
    """Display a combination."""
    combo = wa.combination
    name = COMBO_NAMES.get(combo.type, str(combo.type))
    if combo.type == CombinationType.PASS:
        return "Pass"
    cards = " ".join(card_str(c, high_card_rank) for c in combo.cards)
    extra = ""
    if combo.type == CombinationType.BOMB:
        extra = f" ({combo.bomb_size}-of-a-kind)"
    if combo.type == CombinationType.STRAIGHT_FLUSH:
        extra = " (BOMB)"
    if wa.assignments:
        wc_info = ", ".join(f"{repr(orig)}→{repr(sub)}" for orig, sub in wa.assignments)
        extra += f" [wildcard: {wc_info}]"
    return f"{name}{extra}: {cards}"


# --- Human agent ---

class HumanAgent:
    """Interactive human agent for CLI play."""

    def choose_move(
        self,
        player: Player,
        game_state: GameState,
        moves: list[WildcardAssignment],
    ) -> WildcardAssignment:
        hr = player.high_card_rank
        print()
        print("=" * 60)
        print(f"\033[1mYour turn (Player {player.id})\033[0m")
        print(f"High card rank: {RANK_SYMBOLS[hr]} | Cards: {player.card_count}")
        print()

        # Show game context
        if game_state.active_combo and game_state.active_combo.type != CombinationType.PASS:
            active_wa = WildcardAssignment(game_state.active_combo, ())
            print(f"  Active: {combo_str(active_wa)}")
            print(f"  (played by Player {game_state.trick_leader})")
        else:
            print("  \033[1;32mYou are leading — play anything!\033[0m")

        # Show opponents' card counts
        print()
        for i, count in enumerate(game_state.cards_remaining):
            if i != player.id:
                marker = " (teammate)" if game_state.teams and game_state.teams[i] == player.team else ""
                print(f"  Player {i}: {count} cards{marker}")
        print()

        # Show hand
        sorted_hand = sorted(player.hand, key=lambda c: (effective_rank(c, hr), c.suit))
        print("Your hand:")
        print(hand_str(sorted_hand, hr))
        print()

        # Show legal moves
        print(f"Legal moves ({len(moves)}):")
        for i, m in enumerate(moves):
            print(f"  {i:3d}: {combo_str(m, hr)}")
        print()

        # Get choice
        while True:
            try:
                raw = input("Choose move number (or 'q' to quit): ").strip()
                if raw.lower() == "q":
                    print("Quitting...")
                    sys.exit(0)
                idx = int(raw)
                if 0 <= idx < len(moves):
                    chosen = moves[idx]
                    print(f"  → Playing: {combo_str(chosen, hr)}")
                    return chosen
                print(f"  Invalid: enter 0–{len(moves) - 1}")
            except ValueError:
                print("  Enter a number or 'q'")
            except (EOFError, KeyboardInterrupt):
                print("\nQuitting...")
                sys.exit(0)


# --- Game runner ---

def run_cli_game(
    num_players: int = 3,
    human_player: int = 0,
    num_decks: int | None = None,
    teams: list[int] | None = None,
) -> None:
    """Run an interactive CLI game."""
    agents: list[Agent] = []
    for i in range(num_players):
        if i == human_player:
            agents.append(HumanAgent())
        else:
            agents.append(RandomAgent())

    game = Game(num_players, agents, num_decks=num_decks, teams=teams)
    game.deal()

    print("\033[1;36m" + "=" * 60)
    print("  ZHENG SHANG YOU (争上游)")
    print("=" * 60 + "\033[0m")
    print(f"Players: {num_players} | Decks: {game.num_decks}")
    if teams:
        print(f"Teams: {teams}")
    print(f"You are Player {human_player}")
    print(f"Your high card rank: {RANK_SYMBOLS[game.players[human_player].high_card_rank]}")
    print()

    turn = 0
    while game.phase == GamePhase.PLAYING:
        record = game.play_turn()
        if record is None:
            break

        pid = record.player_id
        combo = record.combination

        # Show AI moves
        if pid != human_player:
            if combo.type == CombinationType.PASS:
                print(f"  Player {pid} passes.")
            else:
                wa = WildcardAssignment(combo, ())
                print(f"  Player {pid} plays: {combo_str(wa)}")

            # Check if someone finished
            if not game.players[pid].has_cards:
                print(f"  \033[1mPlayer {pid} is out of cards!\033[0m")

        turn += 1

    # Game over
    print()
    print("\033[1;36m" + "=" * 60)
    print("  GAME OVER")
    print("=" * 60 + "\033[0m")
    print(f"Finish order: {game.finish_order}")
    print(f"Winner: Player {game.winner}")
    if game.winning_team is not None:
        print(f"Winning team: {game.winning_team}")
    if game.winner == human_player:
        print("\033[1;32mYou win!\033[0m")
    else:
        print("\033[1;31mYou lost.\033[0m")


def main() -> None:
    """Entry point with argument parsing."""
    import argparse
    parser = argparse.ArgumentParser(description="Play Zheng Shang You")
    parser.add_argument("-n", "--players", type=int, default=3, help="Number of players (2-6)")
    parser.add_argument("-p", "--player", type=int, default=0, help="Your player index")
    parser.add_argument("-d", "--decks", type=int, default=None, help="Number of decks")
    parser.add_argument("-t", "--teams", type=int, nargs="*", default=None,
                        help="Team assignments (e.g., 0 1 0 1)")
    args = parser.parse_args()

    run_cli_game(
        num_players=args.players,
        human_player=args.player,
        num_decks=args.decks,
        teams=args.teams,
    )


if __name__ == "__main__":
    main()
