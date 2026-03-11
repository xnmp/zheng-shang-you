"""Tests for card and deck representation."""
from zsy.cards import Suit, Rank, Card, Deck


class TestSuit:
    def test_standard_suits(self):
        assert Suit.HEARTS.value == 0
        assert Suit.DIAMONDS.value == 1
        assert Suit.CLUBS.value == 2
        assert Suit.SPADES.value == 3

    def test_joker_suits(self):
        assert Suit.SMALL_JOKER.value == 4
        assert Suit.BIG_JOKER.value == 5


class TestRank:
    def test_rank_values_ascending(self):
        assert Rank.THREE.value < Rank.FOUR.value
        assert Rank.FOUR.value < Rank.FIVE.value
        assert Rank.TEN.value < Rank.JACK.value
        assert Rank.JACK.value < Rank.QUEEN.value
        assert Rank.QUEEN.value < Rank.KING.value
        assert Rank.KING.value < Rank.ACE.value
        assert Rank.ACE.value < Rank.TWO.value

    def test_joker_rank(self):
        assert Rank.JOKER.value > Rank.TWO.value

    def test_all_ranks_count(self):
        # 3-10, J, Q, K, A, 2, JOKER = 14 ranks
        assert len(Rank) == 14


class TestCard:
    def test_card_creation(self):
        card = Card(Rank.ACE, Suit.SPADES)
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES

    def test_card_equality(self):
        c1 = Card(Rank.THREE, Suit.HEARTS)
        c2 = Card(Rank.THREE, Suit.HEARTS)
        assert c1 == c2

    def test_card_inequality(self):
        c1 = Card(Rank.THREE, Suit.HEARTS)
        c2 = Card(Rank.THREE, Suit.SPADES)
        assert c1 != c2

    def test_card_hashable(self):
        c1 = Card(Rank.THREE, Suit.HEARTS)
        c2 = Card(Rank.THREE, Suit.HEARTS)
        assert hash(c1) == hash(c2)
        assert len({c1, c2}) == 1

    def test_small_joker(self):
        card = Card.small_joker()
        assert card.rank == Rank.JOKER
        assert card.suit == Suit.SMALL_JOKER
        assert card.is_joker

    def test_big_joker(self):
        card = Card.big_joker()
        assert card.rank == Rank.JOKER
        assert card.suit == Suit.BIG_JOKER
        assert card.is_joker

    def test_regular_card_not_joker(self):
        card = Card(Rank.ACE, Suit.SPADES)
        assert not card.is_joker

    def test_card_repr(self):
        card = Card(Rank.ACE, Suit.SPADES)
        r = repr(card)
        assert "A" in r
        assert "♠" in r

    def test_joker_repr(self):
        assert "Small" in repr(Card.small_joker()) or "small" in repr(Card.small_joker()).lower()
        assert "Big" in repr(Card.big_joker()) or "big" in repr(Card.big_joker()).lower()

    def test_card_ordering_by_rank(self):
        three = Card(Rank.THREE, Suit.HEARTS)
        ace = Card(Rank.ACE, Suit.HEARTS)
        assert three < ace

    def test_card_ordering_same_rank_by_suit(self):
        h = Card(Rank.THREE, Suit.HEARTS)
        s = Card(Rank.THREE, Suit.SPADES)
        # Just needs to be consistent, not a specific order
        assert (h < s) or (s < h)

    def test_card_sortable(self):
        cards = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.KING, Suit.CLUBS),
        ]
        sorted_cards = sorted(cards)
        assert sorted_cards[0].rank == Rank.THREE
        assert sorted_cards[-1].rank == Rank.ACE


class TestDeck:
    def test_single_deck_size(self):
        deck = Deck(num_decks=1)
        assert len(deck.cards) == 54

    def test_double_deck_size(self):
        deck = Deck(num_decks=2)
        assert len(deck.cards) == 108

    def test_single_deck_composition(self):
        deck = Deck(num_decks=1)
        # 13 ranks * 4 suits = 52 standard cards + 2 jokers
        jokers = [c for c in deck.cards if c.is_joker]
        standard = [c for c in deck.cards if not c.is_joker]
        assert len(jokers) == 2
        assert len(standard) == 52

    def test_double_deck_composition(self):
        deck = Deck(num_decks=2)
        jokers = [c for c in deck.cards if c.is_joker]
        standard = [c for c in deck.cards if not c.is_joker]
        assert len(jokers) == 4
        assert len(standard) == 104

    def test_single_deck_joker_types(self):
        deck = Deck(num_decks=1)
        jokers = [c for c in deck.cards if c.is_joker]
        small = [j for j in jokers if j.suit == Suit.SMALL_JOKER]
        big = [j for j in jokers if j.suit == Suit.BIG_JOKER]
        assert len(small) == 1
        assert len(big) == 1

    def test_double_deck_joker_types(self):
        deck = Deck(num_decks=2)
        jokers = [c for c in deck.cards if c.is_joker]
        small = [j for j in jokers if j.suit == Suit.SMALL_JOKER]
        big = [j for j in jokers if j.suit == Suit.BIG_JOKER]
        assert len(small) == 2
        assert len(big) == 2

    def test_deal_evenly(self):
        deck = Deck(num_decks=1)
        hands = deck.deal(3)
        assert len(hands) == 3
        assert len(hands[0]) == 18
        assert len(hands[1]) == 18
        assert len(hands[2]) == 18
        # All cards dealt
        all_dealt = [c for h in hands for c in h]
        assert len(all_dealt) == 54

    def test_deal_uneven(self):
        # 54 cards / 4 players = 13 each with 2 remaining
        deck = Deck(num_decks=1)
        hands = deck.deal(4)
        assert len(hands) == 4
        sizes = [len(h) for h in hands]
        # First 2 players get 14, last 2 get 13 (or similar distribution)
        assert sum(sizes) == 54
        assert max(sizes) - min(sizes) <= 1

    def test_deal_double_deck(self):
        deck = Deck(num_decks=2)
        hands = deck.deal(4)
        assert len(hands) == 4
        assert sum(len(h) for h in hands) == 108

    def test_deal_is_shuffled(self):
        """Two deals should (almost certainly) produce different hands."""
        deck1 = Deck(num_decks=1)
        deck2 = Deck(num_decks=1)
        hands1 = deck1.deal(3)
        hands2 = deck2.deal(3)
        # Extremely unlikely to be identical after shuffling
        all1 = [c for h in hands1 for c in h]
        all2 = [c for h in hands2 for c in h]
        assert all1 != all2

    def test_single_deck_all_standard_ranks_present(self):
        deck = Deck(num_decks=1)
        standard = [c for c in deck.cards if not c.is_joker]
        ranks = {c.rank for c in standard}
        expected_ranks = {r for r in Rank if r != Rank.JOKER}
        assert ranks == expected_ranks

    def test_single_deck_each_rank_has_four_suits(self):
        deck = Deck(num_decks=1)
        standard = [c for c in deck.cards if not c.is_joker]
        for rank in Rank:
            if rank == Rank.JOKER:
                continue
            cards_of_rank = [c for c in standard if c.rank == rank]
            assert len(cards_of_rank) == 4
            suits = {c.suit for c in cards_of_rank}
            assert suits == {Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES}
