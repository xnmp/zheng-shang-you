"""Comprehensive edge case tests for the game engine.

Covers: multi-deck scenarios, joker rules, dynamic ranking edge cases,
legal move correctness properties, and integration scenarios.
"""
import random
from collections import Counter

import pytest
from zsy.cards import Card, Rank, Suit, Deck, STANDARD_RANKS
from zsy.combinations import (
    Combination,
    CombinationType,
    classify,
)
from zsy.game import Game, GamePhase
from zsy.agents import RandomAgent
from zsy.legal_moves import legal_moves
from zsy.ranking import effective_rank, is_wildcard
from zsy.player import Player


def c(rank_str: str, suit_str: str = "h") -> Card:
    rank_map = {
        "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE, "6": Rank.SIX,
        "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE, "10": Rank.TEN,
        "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
        "2": Rank.TWO,
    }
    suit_map = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
    return Card(rank_map[rank_str], suit_map[suit_str])


# --- Multi-deck scenarios ---

class TestMultiDeck:
    def test_five_of_a_kind_possible(self):
        # In double deck, can have 5+ of same rank
        cards = [c("8", s) for s in ("h", "d", "c", "s")] + [c("8", "h")]
        combo = classify(cards)
        assert combo is not None
        assert combo.type == CombinationType.BOMB
        assert combo.bomb_size == 5

    def test_double_deck_has_duplicates(self):
        deck = Deck(num_decks=2)
        card_counts = Counter(deck.cards)
        # Each standard card appears exactly twice
        for card in deck.cards:
            if not card.is_joker:
                assert card_counts[card] == 2

    def test_double_deck_four_jokers(self):
        deck = Deck(num_decks=2)
        jokers = [c for c in deck.cards if c.is_joker]
        assert len(jokers) == 4

    def test_six_player_game_completes(self):
        random.seed(42)
        agents = [RandomAgent() for _ in range(6)]
        game = Game(6, agents, num_decks=2)
        winner = game.run()
        assert game.phase == GamePhase.FINISHED


# --- Joker rules ---

class TestJokerRules:
    def test_big_joker_beats_small_joker(self):
        big = classify([Card.big_joker()])
        small = classify([Card.small_joker()])
        assert big.beats(small)
        assert not small.beats(big)

    def test_joker_is_single(self):
        combo = classify([Card.big_joker()])
        assert combo.type == CombinationType.SINGLE

    def test_mixed_joker_pair_invalid(self):
        """A pair of different joker types is not valid."""
        combo = classify([Card.small_joker(), Card.big_joker()])
        assert combo is None

    def test_same_joker_pair_valid_in_double_deck(self):
        combo = classify([Card.small_joker(), Card.small_joker()])
        assert combo is not None
        assert combo.type == CombinationType.PAIR

    def test_joker_not_in_straight(self):
        """Jokers cannot be part of straights."""
        cards = [c("10", "h"), c("J", "d"), c("Q", "c"), c("K", "s"), Card.small_joker()]
        combo = classify(cards)
        assert combo is None

    def test_joker_not_in_bomb(self):
        """Jokers cannot form bombs."""
        cards = [Card.small_joker()] * 4
        combo = classify(cards)
        # 4 small jokers should not be a bomb (jokers don't form N-of-a-kind)
        assert combo is None


# --- Dynamic ranking edge cases ---

class TestDynamicRankingEdgeCases:
    def test_all_ranks_as_supreme(self):
        """Each rank should work as the supreme rank."""
        for high_rank in STANDARD_RANKS:
            card = Card(high_rank, Suit.SPADES)
            ace = Card(Rank.ACE, Suit.SPADES)
            if high_rank == Rank.ACE:
                continue
            assert effective_rank(card, high_rank) > effective_rank(ace, high_rank)

    def test_wildcard_for_each_rank(self):
        """The wildcard is always the heart suit of the supreme rank."""
        for high_rank in STANDARD_RANKS:
            wc = Card(high_rank, Suit.HEARTS)
            assert is_wildcard(wc, high_rank)
            # Other suits are not wildcards
            assert not is_wildcard(Card(high_rank, Suit.SPADES), high_rank)

    def test_advance_through_all_ranks(self):
        """Advancing high card cycles through all ranks."""
        p = Player(id=0)
        seen = set()
        for _ in range(len(STANDARD_RANKS)):
            seen.add(p.high_card_rank)
            p.advance_high_card()
        assert len(seen) == len(STANDARD_RANKS)


# --- Legal move generator properties ---

class TestLegalMoveProperties:
    def test_pass_always_available(self):
        """Pass should always be available."""
        for seed in range(10):
            random.seed(seed)
            deck = Deck(1)
            hand = deck.deal(3)[0]
            moves = legal_moves(hand, Rank.TWO)
            pass_moves = [m for m in moves if m.combination.type == CombinationType.PASS]
            assert len(pass_moves) >= 1

    def test_at_least_one_single_with_nonempty_hand(self):
        """A non-empty hand should always be able to play at least one single."""
        for seed in range(10):
            random.seed(seed)
            deck = Deck(1)
            hand = deck.deal(3)[0]
            moves = legal_moves(hand, Rank.TWO)
            singles = [m for m in moves if m.combination.type == CombinationType.SINGLE]
            assert len(singles) >= 1

    def test_following_moves_all_beat_active(self):
        """When following, every non-pass move should beat the active combo."""
        random.seed(42)
        deck = Deck(1)
        hand = deck.deal(3)[0]
        active = classify([c("5", "h")])
        moves = legal_moves(hand, Rank.TWO, active)
        for m in moves:
            if m.combination.type != CombinationType.PASS:
                assert m.combination.beats(active), f"{m.combination} should beat {active}"

    def test_no_cards_not_in_hand(self):
        """All cards in legal moves should come from the hand (or be substituted)."""
        hand = [c("3", "h"), c("5", "s"), c("5", "d"), c("A", "h")]
        moves = legal_moves(hand, Rank.TWO)
        hand_set = set(hand)
        for m in moves:
            if m.combination.type == CombinationType.PASS:
                continue
            if not m.assignments:
                for card in m.combination.cards:
                    assert card in hand_set, f"{card} not in hand"

    def test_empty_hand_only_pass(self):
        moves = legal_moves([], Rank.TWO)
        assert len(moves) == 1
        assert moves[0].combination.type == CombinationType.PASS


# --- Bomb hierarchy comprehensive ---

class TestBombHierarchyComprehensive:
    def test_all_bomb_sizes_ordered(self):
        """4 < 5 < SF < 6 < 7 < 8 of a kind."""
        four = classify([c("3", s) for s in ("h", "d", "c", "s")])
        five = classify([c("3", s) for s in ("h", "d", "c", "s", "h")])
        sf = classify([c("3", "h"), c("4", "h"), c("5", "h"), c("6", "h"), c("7", "h")])
        six = classify([c("3", s) for s in ("h", "d", "c", "s", "h", "d")])

        assert five.beats(four)
        assert sf.beats(five)
        assert six.beats(sf)
        assert not four.beats(five)
        assert not five.beats(sf)
        assert not sf.beats(six)

    def test_bomb_beats_all_non_bomb_types(self):
        bomb = classify([c("3", "h"), c("3", "d"), c("3", "c"), c("3", "s")])
        non_bombs = [
            classify([c("A")]),  # single
            classify([c("A", "h"), c("A", "s")]),  # pair
            classify([c("A", "h"), c("A", "d"), c("A", "c")]),  # triple
            classify([c("A", "h"), c("A", "d"), c("A", "c"), c("K", "h"), c("K", "s")]),  # full house
            classify([c("10", "h"), c("J", "d"), c("Q", "c"), c("K", "s"), c("A", "h")]),  # straight
        ]
        for nb in non_bombs:
            assert bomb.beats(nb), f"Bomb should beat {nb.type}"

    def test_same_size_bomb_higher_rank_wins(self):
        low = classify([c("3", s) for s in ("h", "d", "c", "s")])
        high = classify([c("A", s) for s in ("h", "d", "c", "s")])
        assert high.beats(low)
        assert not low.beats(high)

    def test_same_rank_same_size_does_not_beat(self):
        a = classify([c("5", s) for s in ("h", "d", "c", "s")])
        b = classify([c("5", s) for s in ("h", "d", "c", "s")])
        assert not a.beats(b)


# --- Integration: full game scenarios ---

class TestIntegrationScenarios:
    def test_game_with_high_card_3(self):
        """Game where a player's high card is 3."""
        random.seed(42)
        agents = [RandomAgent() for _ in range(3)]
        game = Game(3, agents)
        game.players[0].high_card_rank = Rank.THREE
        game.deal()
        winner = game.run()
        assert game.phase == GamePhase.FINISHED

    def test_many_games_no_crash(self):
        """Run 50 games across configurations to check for crashes."""
        configs = [
            (2, 1, None),
            (3, 1, None),
            (4, 2, [0, 1, 0, 1]),
            (5, 2, None),
            (6, 2, [0, 1, 2, 0, 1, 2]),
        ]
        for n, decks, teams in configs:
            for seed in range(10):
                random.seed(seed)
                agents = [RandomAgent() for _ in range(n)]
                game = Game(n, agents, num_decks=decks, teams=teams)
                winner = game.run()
                assert game.phase == GamePhase.FINISHED
                assert winner is not None
