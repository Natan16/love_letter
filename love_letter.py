import random
from random import shuffle 
from state import State
#don't consider the rounds independent


class Player :    
     
    def __init__(self , ID) :
        self.id = ID
        self.hand = []
         
        self.private = [[0 ,0, 0, 0],[0, 0, 0, 0],[0, 0, 0 ,0],[0, 0 ,0 ,0]]

        self.input = []
        self.encoding = []
        self.round_input = []
        self.round_encoding = []
        
    def play(self, logits , state) :
        
        card , target , guess , encoding = self.choose_play( logits , state)
        if self.hand[0] == card :
            self.hand.pop(0)
        else :
            self.hand.pop()
        return card , target , guess , encoding

    def choose_play(self , logits , state) :
        ''' RANDOM PLAYER
        target = self
        guess = None
        #chooses randomly among the possible plays
        if (self.hand[0] == 7 and self.hand[1] >= 5):
            card_index = 0
        elif (self.hand[1] == 7 and self.hand[0] >= 5):
            card_index = 1
        elif self.hand[0] == 8 :
            card_index = 1
        elif self.hand[1] == 8:
            card_index = 0
        else :
            card_index = random.randrange(2)


        if self.hand[card_index] in [1,2,3,6] :
            if( self.hand[card_index] == 1 ) :
                guess = random.randrange(1,9)
            target = random.choice(players[0:self.id] + players[self.id + 1 :]) #random player, except himself 
        if self.hand[card_index] == 5 :
            target = random.choice(players)
        return card_index , target , guess  '''
        #print(logits)
        log_ind = []
        for i in range(len(logits)):
            log_ind.append([logits[i] , i])
        #log_ind = [logits , [x for x in range(len(logits))]]
        log_ind = sorted(log_ind , key = lambda x : x[0] , reverse = True)  
        #print(log_ind)
        for _ , ind in log_ind :
            #print(ind)
            #print(len(logits) - ind - 1)
            card , target , guess = state.outputToPlay([0]*int(ind) + [1] + [0]*(int((len(logits) - ind - 1)))) 
            if state.isValid(card , target , guess) :
                
                encoding = [0]*int(ind) + [1] + [0]*(int(len(logits) - ind - 1))
                #print(encoding)
                #print(card)
                #print(target)
                #print(guess)
                return card , target , guess , encoding
        #print("no valid play possible")
        return None, None , None , None
                
class RandomPlayer(Player) :    
   
    def choose_play(self , logits , state) :
        #RANDOM PLAYER
        target = self
        guess = None
        #chooses randomly among the possible plays
        if (self.hand[0] == 7 and self.hand[1] >= 5):
            card_index = 0
        elif (self.hand[1] == 7 and self.hand[0] >= 5):
            card_index = 1
        elif self.hand[0] == 8 :
            card_index = 1
        elif self.hand[1] == 8:
            card_index = 0
        else :
            card_index = random.randrange(2)


        if self.hand[card_index] in [1,2,3,6] :
            if( self.hand[card_index] == 1 ) :
                guess = random.randrange(1,9)
            target = random.choice(players[0:self.id] + players[self.id + 1 :]) #random player, except himself 
        if self.hand[card_index] == 5 :
            target = random.choice(players)
        return card_index , target , guess , None #no encoding for now
        

def generateTrainingData( NUMBER_OF_PLAYERS , NUMBER_OF_ITERATIONS , network):
    
    N = NUMBER_OF_PLAYERS
    WIN = (13 - 1)//N + 1 #number of rounds a player has to win 

    train = []
    labels = []
    
    while len(train) < 1.2*NUMBER_OF_ITERATIONS :

        players = [Player(i) for i in range(N)]
        state = State(N)
                                                
        #plays round , returns winner of the round
        def play_round(i = random.randrange(N)) :
            deck = [8,7,6,5,5,4,4,3,3,2,2,1,1,1,1,1]
            shuffle(deck) #shuffle the deck
            removed = deck.pop() #leaves one card out
            #print(state.eliminated)
            state.reset()
            #print(state.eliminated)
            #each player draws a card
            for player in players :
                player.hand = [deck.pop()]
                
            inRound = N #players still in the current round
            while len(deck) > 0 and inRound > 1 :
                #print(state.eliminated)
                
                if not state.eliminated[0] :
                    
                    player = players[i]
                    #imunity wears off
                    state.imunity[0] = False

                    #draws a card
                    player.hand.append(deck.pop())
                    state.hand = list(sorted(players[i].hand))
                    #print(player.hand)
                    #decides wich card to play
                    inp = state.stateToInput()
                    logits = network.getLogits(inp)
                    card , target_index , guess , encoding =  players[i].play(logits , state)
                    #print(card)
                    #stores possible training data
                    player.input.append(inp)
                    player.encoding.append(encoding)
                    player.round_input.append(inp)
                    player.round_encoding.append(encoding)
                    
                    
                    #updates tot_sum and cards_played            
                    state.cards_played[card - 1] -= 1
                    state.tot_sum[0] += card
                                                
                    #flushes public and private data
                    #if he played a card that he could have before the public data vanishes
                    if state.public[0][card - 1] == 0 :
                        state.public[0] = [0]*8
                    #if he played the card the he had before, the private data about him vanishes
                    if card == player.hand[0] :
                        for p in players :
                            p.private[(player.id - p.id + N)%N] = [0, 0, 0, 0]        
                                
                    
                    target = players[(target_index + player.id)%N]
                     
                    if card == 1 and not state.imunity[target_index]:
                        result = guess == target.hand[0]
                        if result :
                            state.eliminated[target_index] = True
                            state.public[target_index] = [1]*8
                            state.public[target_index][player.hand[0] - 1] = 0                    
                            inRound -= 1
                            
                        else :
                            state.public[target_index][guess - 1] = 1#target doesn't have the card guessed
                    
                    elif card == 2 and not state.imunity[target_index]:
                        #private knowledge changes                        
                        for p in players :
                            t = (target_index - ( p.id - player.id + N )%N + N)%N
                            p0 = (player.id - p.id + N)%N
                            if p.id == player.id :
                                p.private[t][p0] = target.hand[0]
                            else :
                                p.private[t][p0] = 1
 
                         
                    elif card == 3 and not state.imunity[target_index]:
                        if player.hand[0] > target.hand[0] :
                            state.eliminated[target_index] =  True
                            state.public[target_index] = [1]*8
                            state.public[target_index][player.hand[0] - 1] = 0                    
                            inRound -= 1
                                                
                            state.public[target_index]
                            state.public[0] = [1]*target.hand[0] + state.public[0][target.hand[0]:]
                            
                        elif player.hand[0] < target.hand[0] :
                            state.eliminated[0] = True
                            state.public[0] = [1]*8
                            state.public[0][player.hand[0] - 1] = 0
                            inRound -= 1
                                                
                            state.public[target_index] = [1]*player.hand[0] + state.public[target_index][player.hand[0]:]
                            
                        else :
                            #private knowledge of both changes        
                            for p in players :
                                t = (target_index - ( p.id - player.id + N )%N + N)%N
                                p0 = (player.id - p.id + N)%N
                               
                                p.private[t][p0] = target.hand[0] if p.id == player.id else 1
                                p.private[p0][t] = player.hand[0] if p.id == player.id else 1
                            
                            
                    elif card == 4 :
                        state.imunity[0] = True
                        
                    elif card == 5 and not state.imunity[target_index]:
                        card_discarded = target.hand.pop()
                        state.cards_played[card_discarded - 1] -= 1
                        state.tot_sum[target_index] += card_discarded
                        if card_discarded == 8 :
                            state.eliminated[target_index] = True
                            state.public[target_index] = [1,1,1,1,1,1,1,0]
                            inRound -= 1
                        elif len(deck) :
                            target.hand.append(deck.pop())
                        else :
                            target.hand.append(removed)

                        #flushes public knowledge about target
                        state.public[target_index] = [0]*8
                        state.countess[0] = 0

                        
                    
                    elif card == 6 and not state.imunity[target_index] :
                        target.hand , player.hand = player.hand , target.hand
                        #private knowledge changes                        
                        for p in players :
                            t = (target_index - ( p.id - player.id + N )%N + N)%N
                            p0 = (player.id - p.id + N)%N
                            p.private[t][p0] = target.hand[0] if p.id == player.id else 1
                            p.private[p0][t] = player.hand[0] if p.id == player.id else 1
                        #public knowledge also changes
                        state.public[0] , state.public[target_index] = state.public[target_index] , state.public[0]
                        state.countess[0] = 0 

                    elif card == 7 :
                        state.countess[0] = 1
                        
                    elif card == 8 :
                        state.eliminated[0] = True
                        state.public[0] = [1]*8
                        state.public[0][player.hand[0] - 1] = 0
                        inRound -= 1 
                    
                if len(deck) > 0 and inRound > 1 :
                    i = (i+1)%N
                    state.nextPlayer(players[i])
                    
            
            
              
            #the remaning player with the hightest card in the hand, wins the round
            hightest = 0
            possible_winners = [] 
            for p in players :
                if not state.eliminated[(p.id - player.id + N)%N] :
                    possible_winners.append(p) 
            #possible_winners = list(filter(lambda x : not x.eliminated , players))
            for p in possible_winners :
                if p.hand[0] > hightest :
                    hightest = p.hand[0]
            
            possible_winners = list(filter(lambda x : x.hand[0] == hightest , possible_winners ))
            
            if len(possible_winners) == 1 :
                winner = possible_winners[0]
            else :
                
                hightest = 0
                for pw in possible_winners :
                    if state.tot_sum[(pw.id - player.id + N)%N] > hightest :
                        hightest = state.tot_sum[(pw.id - player.id + N)%N]    
                possible_winners_aux = []
                for pw in possible_winners :
                    if state.tot_sum[(pw.id - player.id + N)%N] == hightest :
                        possible_winners_aux.append(pw)
                possible_winners = possible_winners_aux
                winner = random.choice(possible_winners)
            winner_index = (winner.id - i + N)%N
            state.victories[winner_index] += 1
            #so the state if placed correctly for the next round
            if winner_index != 0 :
                for _ in range(winner_index) :
                    player = players[(player.id + 1)%N]
                    state.nextPlayer(player)
            return winner

        #play rounds until a winner is found
        winner = play_round()
        train += winner.round_input
        labels += winner.round_encoding
        for p in players :
            p.round_input = []
            p.round_encoding = []
        #print (winner.id) 
        while WIN not in state.victories:
            winner = play_round(winner.id)
            train += winner.round_input
            labels += winner.round_encoding
            for p in players :
                p.round_input = []
                p.round_encoding = []
        
            #print (winner.id)     
        #guarda as duplas (input , 1-hot ecoding) tomadas pelo jogador que ganhou a partida
        train += winner.input
        labels += winner.encoding
        print(len(train))
        print ("----------------")
    return train[:NUMBER_OF_ITERATIONS] , labels[:NUMBER_OF_ITERATIONS] , train[NUMBER_OF_ITERATIONS + 1: int(1.1*NUMBER_OF_ITERATIONS)] ,labels[NUMBER_OF_ITERATIONS + 1: int(1.1*NUMBER_OF_ITERATIONS)] , train[int(1.1*NUMBER_OF_ITERATIONS) + 1: ] ,labels[int(1.1*NUMBER_OF_ITERATIONS) + 1: ]

