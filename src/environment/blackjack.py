import numpy as np

class BlackjackEnvironment:
    def __init__(self, verbose=False):
        """
        Initialize the Blackjack environment.
        Args:
            verbose (bool): If True, print additional information during the game.
        """
        # Define the deck
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Ace, 2-10, Jack, Queen
        self.card_value = {}
        self.verbose = verbose

        # Assign values to cards in the deck
        for c in self.deck:
            if c == 1:
                self.card_value[c] = [c, 11]  # Ace can be 1 or 11
            elif c in np.arange(2, 11):
                self.card_value[c] = c # Cards 2-10 have their face values
            else:
                self.card_value[c] = 10 # Face cards (Jack, Queen) have a value of 10

    def deal_card(self):
        """
        Simulate dealing a card from the deck.

        Returns:
            int: A randomly chosen card from the deck.
        """
        # Simulate dealing a card from the deck.
        return np.random.choice(self.deck)

    def card_mapper(self, card_hand):
        """
        Map cards in a hand to their corresponding values.

        Args:
            card_hand (list of int): A list of cards in the hand.

        Returns:
            list of int: A list of card values.
        """
        # Map cards in a hand to their corresponding values
        hand = []
        for card in card_hand:
            hand.append(self.card_value[card])
        return hand

    def initial_state(self):
        """
        Initialize the initial state of the game with player and dealer hands.

        Returns:
            tuple of lists: Player and dealer hands as lists.
        """
        player_hand = [self.deal_card(), self.deal_card()]
        #dealer_hand = [self.deal_card(), self.deal_card()]
        dealer_hand = [self.deal_card()]
        return player_hand, dealer_hand

    def player_action(self, player_hand):
        # The player can choose to hit or stand
        return np.random.choice(['hit', 'stand'])

    def hand_total_original(self, hand):
        """
        Calculate the total value of a hand, considering possible values for Aces.

        Args:
            hand (list of int): A list of card values in the hand.

        Returns:
            Total values with and without considering Aces, and a flag for the presence of Ace.
        """
        mapped_hand = self.card_mapper(hand)
        combo_1 = 0
        combo_2 = 0
        has_ace = False
        for card in mapped_hand:
            if card == [1, 11]:
                combo_1 += 1
                combo_2 += 11
                has_ace = True
            else:
                combo_1 += card
                combo_2 += card
        return combo_1, combo_2, has_ace

    def hand_total(self, hand):
        """
        Calculate the total value of a hand, considering possible values for Aces.

        Args:
            hand (list of int): A list of card values in the hand.

        Returns:
            Tuple containing the soft (with Ace as 11) and hard (with Ace as 1) hand totals and a flag for the presence of Ace.
        """
        mapped_hand = self.card_mapper(hand)
        soft_total = 0
        hard_total = 0
        has_ace = False

        for card in mapped_hand:
            if card == [1, 11]:
                if soft_total + 11 <= 21:
                    soft_total += 11  # Ace as 11 in soft total
                    hard_total += 1  # Ace as 1 in hard total
                else:
                    soft_total += 1  # Ace as 1 in soft total
                    hard_total += 1  # Ace as 1 in hard total
                has_ace = True
            else:
                soft_total += card
                hard_total += card

        return hard_total, soft_total, has_ace

    def dealer_action(self, dealer_hand):
        """
        Determine the dealer's action (hit or stand).

        Args:
            dealer_hand (list of int): Dealer's current hand.

        Returns:
            str: The selected action, either 'hit' or 'stand'.
        """
        combo_1, combo_2, has_ace = self.hand_total(dealer_hand)

        hard_16 = False
        soft_17 = False
        lower = False

        if combo_1 == 16 and has_ace == False:  # hard 16
            hard_16 = True;
        elif combo_2 == 17 and has_ace:  # soft 17
            soft_17 = True ;
        elif min(combo_1, combo_2) <= 16 and hard_16 == False and soft_17 == False: # lower value
            lower = True;
        else:
            pass

        if hard_16 or soft_17 or lower:
            return "hit"
        else:
            return "stand"

    def accepted_value_original(self, player_values, dealer_values):
        """
        Determine the accepted values for both player and dealer.

        Args:
            player_values (list of int): Possible total values for the player.
            dealer_values (list of int): Possible total values for the dealer.

        Returns:
            list of int and list of int: Accepted values for the player and dealer, considering values not exceeding 21.
        """
        # Return list of values in the player's and dealer's hands that are less than or equal to 21
        return list(set(v for v in player_values if v <= 21)), list(set(d for d in dealer_values if d <= 21))

    def accepted_value(self, cards):
        """
        Determine the accepted values for both player and dealer.

        Args:
            cards (list of int): Possible total values for the cards.

        Returns:
            list of int and list of int: Accepted values for the player and dealer, considering values not exceeding 21.
        """
        # Return list of values in the player's and dealer's hands that are less than or equal to 21
        return list(set(v for v in cards if v <= 21))

    def step(self, player_hand, dealer_hand, player_action):
        """
        Simulate a single step in the game.

        Args:
            player_hand (list of int): Player's current hand.
            dealer_hand (list of int): Dealer's current hand.
            player_action (str): The player's chosen action, either 'hit' or 'stand'.

        Returns:
            tuple: New player hand, dealer hand, reward, and a flag indicating if the game is done.
        """
        player_combo_1, player_combo_2, _ = self.hand_total(player_hand)
        dealer_combo_1, dealer_combo_2, _ = self.hand_total(dealer_hand)
        dealer_max = max(dealer_combo_1, dealer_combo_2)
        dealer_min = min(dealer_combo_1, dealer_combo_2)
        player_max = max(player_combo_1, player_combo_2)
        player_min = min(player_combo_1, player_combo_2)

        # Player has a blackjack, and the dealer does not
        if (player_max == 21 or player_min == 21) and len(player_hand) == 2 and (dealer_max != 21 and dealer_min != 21):
            if self.verbose and count / 2**power == 1:
                print(f"Player:\n---Hand: {player_hand}\n---Max Value: {player_max}\n---Min Value: {player_min}")
                print(f"Player Action: stand")
                print("Player wins (BlackJack).\nRewards: +1.5\n")
            return player_hand, dealer_hand, 1.5, True  # Player wins; black jack

        # Both player and dealer have blackjack
        elif (player_max == 21 or player_min == 21) and len(player_hand) == 2 and (dealer_max == 21 or dealer_min == 21):
            if self.verbose and count / 2**power == 1:
                print(f"Player:\n---Hand: {player_hand}\n---Max Value: {player_max}\n---Min Value: {player_min}")
                print(f"Player Action: stand")
                print("Tie (Both BlackJack).\nRewards: 0\n")
            return player_hand, dealer_hand, 0, True # Tie

        else:
            # Player doesn't have a blackjack
            if self.verbose and count / 2**power == 1:
                print(f"Player:\n---Hand: {player_hand}\n---Max Value: {player_max}\n---Min Value: {player_min}")
                print(f"Player Action: {player_action}")
            # Player chooses to hit
            if player_action == 'hit':
                player_hand.append(self.deal_card())
                player_combo_1, player_combo_2, _ = self.hand_total(player_hand)
                player_max = max(player_combo_1, player_combo_2)
                player_min = min(player_combo_1, player_combo_2)

                # Player busts, loses
                if player_min > 21:
                    if self.verbose and count / 2**power == 1:
                        print(f"Player:\n---Hand: {player_hand}\n---Max Value: {player_max}\n---Min Value: {player_min}")
                        print("Player loses (Busts).\nRewards: -1\n")
                    return player_hand, dealer_hand, -1, True

                else:
                    return player_hand, dealer_hand, 0, False
            else:
                # Player stands, and it's the dealer's turn
                dealer_action = self.dealer_action(dealer_hand)
                while True:
                    dealer_action = self.dealer_action(dealer_hand)
                    # Dealer chooses to hit
                    if dealer_action == 'hit':
                        if self.verbose and count / 2**power == 1:
                            print(f"Dealer:\n---Hand: {dealer_hand}\n---Max Value: {dealer_max}\n---Min Value: {dealer_min}")
                            print(f"Dealer Action: {dealer_action}")
                        dealer_hand.append(self.deal_card())
                        dealer_combo_1, dealer_combo_2, _ = self.hand_total(dealer_hand)
                        dealer_max = max(dealer_combo_1, dealer_combo_2)
                        dealer_min = min(dealer_combo_1, dealer_combo_2)
                    # Dealer stands
                    else:
                        #player_accepted_val, dealer_accepted_val = self.accepted_value(player_values=[player_combo_1, player_combo_2], dealer_values=[dealer_combo_1, dealer_combo_2])
                        player_accepted_val = self.accepted_value([player_combo_1, player_combo_2])
                        dealer_accepted_val = self.accepted_value([dealer_combo_1, dealer_combo_2])

                        if self.verbose and count / 2**power == 1:
                            print(f"Dealer:\n---Hand: {dealer_hand}\n---Max Value: {dealer_max}\n---Min Value: {dealer_min}")
                            print(f"Dealer Action: {dealer_action}")
                        # Dealer busts, player wins
                        if dealer_min > 21:
                            if self.verbose and count / 2**power == 1:
                                print("Player wins (Dealer Busts).\nRewards: +1\n")
                            return player_hand, dealer_hand, 1, True
                        # Player wins, has a higher total
                        elif len(player_accepted_val)>0 and len(dealer_accepted_val)>0 and max(player_accepted_val) > max(dealer_accepted_val):
                            if self.verbose and count / 2**power == 1:
                                print("Player wins (Better Hand).\nRewards: +1\n")
                            return player_hand, dealer_hand, 1, True
                         # Player wins
                        elif len(player_accepted_val)>0 and len(dealer_accepted_val)==0:
                            if self.verbose and count / 2**power == 1:
                                print("Player wins (Better Hand).\nRewards: +1\n")
                            return player_hand, dealer_hand, 1, True
                        # Tie game
                        elif len(player_accepted_val)>0 and len(dealer_accepted_val)>0 and max(player_accepted_val) == max(dealer_accepted_val):
                            if self.verbose and count / 2**power == 1:
                                print("Tie.\nRewards: 0\n")
                            return player_hand, dealer_hand, 0, True
                        # Player loses
                        else:
                            if self.verbose and count / 2**power == 1:
                                print("Player loses (Worse Hand).\nRewards: -1\n")
                            return player_hand, dealer_hand, -1, True

    def play(self):
        """
        Simulate a complete game and return the final reward.

        Returns:
            float: The reward for the game.
        """
         # Simulate a complete game and return the final reward
        player_hand, dealer_hand = self.initial_state()
        done = False
        while not done:
            player_action = self.player_action(player_hand)
            player_hand, dealer_hand, reward, done = self.step(player_hand, dealer_hand, player_action)
            #print(player_hand, dealer_hand, reward, done)
        return reward
