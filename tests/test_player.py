"""Tests for player state."""
from zsy.cards import Card, Rank, Suit
from zsy.player import Player
import pytest


class TestPlayer:
    def test_default_high_card(self):
        p = Player(id=0)
        assert p.high_card_rank == Rank.TWO

    def test_card_count(self):
        p = Player(id=0, hand=[Card(Rank.THREE, Suit.HEARTS), Card(Rank.FOUR, Suit.SPADES)])
        assert p.card_count == 2

    def test_has_cards(self):
        p = Player(id=0, hand=[Card(Rank.THREE, Suit.HEARTS)])
        assert p.has_cards
        p.hand = []
        assert not p.has_cards

    def test_remove_cards(self):
        cards = [Card(Rank.THREE, Suit.HEARTS), Card(Rank.FOUR, Suit.SPADES), Card(Rank.FIVE, Suit.CLUBS)]
        p = Player(id=0, hand=list(cards))
        p.remove_cards([cards[0], cards[2]])
        assert p.hand == [cards[1]]

    def test_remove_cards_not_in_hand(self):
        p = Player(id=0, hand=[Card(Rank.THREE, Suit.HEARTS)])
        with pytest.raises(ValueError):
            p.remove_cards([Card(Rank.ACE, Suit.SPADES)])

    def test_remove_duplicate_cards_from_double_deck(self):
        c = Card(Rank.THREE, Suit.HEARTS)
        p = Player(id=0, hand=[c, c, Card(Rank.FOUR, Suit.SPADES)])
        p.remove_cards([c])
        assert p.hand == [c, Card(Rank.FOUR, Suit.SPADES)]

    def test_advance_high_card(self):
        p = Player(id=0)
        assert p.high_card_rank == Rank.TWO
        p.advance_high_card()
        assert p.high_card_rank == Rank.THREE
        p.advance_high_card()
        assert p.high_card_rank == Rank.FOUR

    def test_advance_high_card_wraps(self):
        p = Player(id=0, high_card_rank=Rank.TWO)
        p.advance_high_card()
        assert p.high_card_rank == Rank.THREE

    def test_team_assignment(self):
        p = Player(id=0, team=1)
        assert p.team == 1

    def test_team_default_none(self):
        p = Player(id=0)
        assert p.team is None
