"""Tests for dynamic card ranking system."""
from zsy.cards import Card, Rank, Suit
from zsy.ranking import effective_rank, is_wildcard, get_wildcards, is_high_card, compare_cards


class TestEffectiveRank:
    def test_big_joker_highest(self):
        bj = Card.big_joker()
        assert effective_rank(bj, Rank.TWO) > effective_rank(Card.small_joker(), Rank.TWO)

    def test_small_joker_above_supreme(self):
        sj = Card.small_joker()
        supreme = Card(Rank.TWO, Suit.SPADES)
        assert effective_rank(sj, Rank.TWO) > effective_rank(supreme, Rank.TWO)

    def test_supreme_rank_above_ace(self):
        # Default high card is 2, so 2 should outrank A
        two = Card(Rank.TWO, Suit.SPADES)
        ace = Card(Rank.ACE, Suit.SPADES)
        assert effective_rank(two, Rank.TWO) > effective_rank(ace, Rank.TWO)

    def test_supreme_rank_changes(self):
        # If high card is 5, then 5 outranks everything except jokers
        five = Card(Rank.FIVE, Suit.SPADES)
        ace = Card(Rank.ACE, Suit.SPADES)
        two = Card(Rank.TWO, Suit.SPADES)
        assert effective_rank(five, Rank.FIVE) > effective_rank(ace, Rank.FIVE)
        assert effective_rank(five, Rank.FIVE) > effective_rank(two, Rank.FIVE)

    def test_non_supreme_normal_ordering(self):
        # With high card = 2, normal cards follow 3 < 4 < ... < A
        three = Card(Rank.THREE, Suit.HEARTS)
        king = Card(Rank.KING, Suit.HEARTS)
        assert effective_rank(three, Rank.TWO) < effective_rank(king, Rank.TWO)

    def test_different_players_different_rankings(self):
        five = Card(Rank.FIVE, Suit.CLUBS)
        seven = Card(Rank.SEVEN, Suit.CLUBS)
        # Player with high card = 5: five is supreme
        assert effective_rank(five, Rank.FIVE) > effective_rank(seven, Rank.FIVE)
        # Player with high card = 2: five is just a normal 5
        assert effective_rank(five, Rank.TWO) < effective_rank(seven, Rank.TWO)

    def test_supreme_all_suits_same_effective_rank(self):
        h = Card(Rank.TWO, Suit.HEARTS)
        s = Card(Rank.TWO, Suit.SPADES)
        d = Card(Rank.TWO, Suit.DIAMONDS)
        c = Card(Rank.TWO, Suit.CLUBS)
        ranks = {effective_rank(x, Rank.TWO) for x in [h, s, d, c]}
        assert len(ranks) == 1  # All have the same effective rank


class TestWildcard:
    def test_heart_suit_high_card_is_wildcard(self):
        card = Card(Rank.TWO, Suit.HEARTS)
        assert is_wildcard(card, Rank.TWO)

    def test_other_suit_high_card_not_wildcard(self):
        card = Card(Rank.TWO, Suit.SPADES)
        assert not is_wildcard(card, Rank.TWO)

    def test_heart_of_non_high_rank_not_wildcard(self):
        card = Card(Rank.THREE, Suit.HEARTS)
        assert not is_wildcard(card, Rank.TWO)

    def test_wildcard_changes_with_rank(self):
        card = Card(Rank.FIVE, Suit.HEARTS)
        assert is_wildcard(card, Rank.FIVE)
        assert not is_wildcard(card, Rank.TWO)

    def test_joker_not_wildcard(self):
        assert not is_wildcard(Card.small_joker(), Rank.TWO)
        assert not is_wildcard(Card.big_joker(), Rank.TWO)


class TestGetWildcards:
    def test_finds_wildcards_in_hand(self):
        hand = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.TWO, Suit.SPADES),
        ]
        wc = get_wildcards(hand, Rank.TWO)
        assert len(wc) == 1
        assert wc[0] == Card(Rank.TWO, Suit.HEARTS)

    def test_no_wildcards(self):
        hand = [Card(Rank.THREE, Suit.HEARTS), Card(Rank.FOUR, Suit.SPADES)]
        assert get_wildcards(hand, Rank.TWO) == []

    def test_double_deck_two_wildcards(self):
        hand = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES),
        ]
        wc = get_wildcards(hand, Rank.TWO)
        assert len(wc) == 2


class TestIsHighCard:
    def test_high_card_any_suit(self):
        for suit in (Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES):
            assert is_high_card(Card(Rank.TWO, suit), Rank.TWO)

    def test_non_high_card(self):
        assert not is_high_card(Card(Rank.THREE, Suit.HEARTS), Rank.TWO)

    def test_joker_not_high_card(self):
        assert not is_high_card(Card.big_joker(), Rank.TWO)


class TestCompareCards:
    def test_higher_beats_lower(self):
        assert compare_cards(
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.THREE, Suit.HEARTS),
            Rank.TWO,
        ) > 0

    def test_supreme_beats_ace(self):
        assert compare_cards(
            Card(Rank.TWO, Suit.SPADES),
            Card(Rank.ACE, Suit.SPADES),
            Rank.TWO,
        ) > 0

    def test_equal_rank(self):
        assert compare_cards(
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Rank.TWO,
        ) == 0
