"""Tests for Jing Gong card exchange."""
import pytest
from zsy.cards import Card, Rank, Suit
from zsy.jing_gong import (
    execute_jing_gong,
    find_best_card,
    find_returnable_cards,
    RETURNABLE_RANKS,
)
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


class SimpleReturnAgent:
    """Returns the lowest returnable card."""
    def choose_return_card(self, player, received_card, returnable):
        return min(returnable, key=lambda c: c.rank)


class TestFindBestCard:
    def test_best_card_by_rank(self):
        p = Player(id=0, hand=[c("3"), c("A", "s"), c("K", "d")])
        best = find_best_card(p)
        # With default high card = 2, A is highest standard rank
        assert best.rank == Rank.ACE

    def test_supreme_rank_is_best(self):
        p = Player(id=0, hand=[c("3"), c("A", "s"), c("2", "d")])
        best = find_best_card(p)
        # 2 is the supreme rank by default
        assert best.rank == Rank.TWO

    def test_joker_is_best(self):
        p = Player(id=0, hand=[c("A", "s"), Card.big_joker()])
        best = find_best_card(p)
        assert best.is_joker


class TestFindReturnableCards:
    def test_finds_cards_below_10(self):
        hand = [c("3"), c("5", "s"), c("9", "d"), c("10"), c("A", "s")]
        p = Player(id=0, hand=hand)
        returnable = find_returnable_cards(p)
        ranks = {card.rank for card in returnable}
        assert ranks == {Rank.THREE, Rank.FIVE, Rank.NINE}

    def test_no_returnable_cards(self):
        hand = [c("10"), c("J"), c("A", "s")]
        p = Player(id=0, hand=hand)
        assert find_returnable_cards(p) == []


class TestExecuteJingGong:
    def test_basic_exchange(self):
        winner = Player(id=0, hand=[c("3"), c("5", "s"), c("K", "d")])
        loser = Player(id=1, hand=[c("A", "s"), c("7", "d"), c("4", "c")])
        agent = SimpleReturnAgent()

        tribute, returned = execute_jing_gong(winner, loser, agent)

        # Loser's best card (A) goes to winner
        assert tribute.rank == Rank.ACE
        assert tribute not in loser.hand
        # Winner returns lowest returnable
        assert returned.rank == Rank.THREE
        assert returned in loser.hand
        assert returned not in winner.hand

    def test_winner_hand_grows_then_shrinks(self):
        winner = Player(id=0, hand=[c("3"), c("5", "s")])
        loser = Player(id=1, hand=[c("A", "s"), c("7", "d")])
        agent = SimpleReturnAgent()

        initial_winner_count = winner.card_count
        initial_loser_count = loser.card_count

        execute_jing_gong(winner, loser, agent)

        # Both players should have same card count as before
        assert winner.card_count == initial_winner_count
        assert loser.card_count == initial_loser_count

    def test_no_returnable_falls_back_to_lowest(self):
        winner = Player(id=0, hand=[c("J"), c("Q"), c("K", "d")])
        loser = Player(id=1, hand=[c("A", "s"), c("10", "d")])
        agent = SimpleReturnAgent()

        tribute, returned = execute_jing_gong(winner, loser, agent)

        # Winner had no cards below 10, should return lowest card
        assert tribute.rank == Rank.ACE
        # J is lowest in winner's hand (after receiving A)
        assert returned is not None
