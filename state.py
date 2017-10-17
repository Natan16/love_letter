from collections import deque 


class State :
    MAX_PLAYERS = 4
    MAX_SUM = 39
    FREQUENCY = [5, 2 ,2 ,2 ,2 ,1, 1 ,1]
    def __init__(self, N) :
        #MAX_PLAYERS = 4
        #MAX_SUM = 39
        #FREQUENCY = [5, 2 ,2 ,2 ,2 ,1, 1 ,1]
        #number of players
        self.N = N
        #model 2 player games as game in wich 2 players are out

        #countess discarded and (prince or king) not discarded latter
        self.countess = deque([0]*N)
        #players have the same card
        #self.same = [[0 0 0 0],[0 0 0 0],[0 0 0 0],[0 0 0 0]]
        
        #soma de cada jogador
        self.tot_sum = deque([0]*N)
        #cartas já jogadas ( poderia colocar cartas restantes )
        self.cards_played = [5, 2 ,2 ,2 ,2 ,1, 1 ,1]
        #jogador já eliminado
        self.eliminated = deque([0]*N)    
        #vitórias de cada jogador
        self.victories = deque([0]*N)
        #imunidade de cada jogadaor
        self.imunity = deque([0]*N)
        #cartas na mão do jogador 0
        self.hand = [0 ,0] #hand of the current player -> Logistic won't work
        #vetor de conhecimento público
        self.public = deque([[0, 0 ,0 ,0 ,0 ,0, 0, 0]]*N) 
        #matriz de conhecimento privado
        #self.private = deque()
        #for _ in range(N) :
        #    self.private.append(deque([0]*N))
        self.private = [[1 ,0 ,0, 0],[0 ,1, 0, 0],[0, 0 ,1 ,0],[0, 0 ,0 ,1]] #private knowledge of current player -> store inside player
        self.private = self.private[:N] + [[0 ,0 ,0 ,0]]*(State.MAX_PLAYERS - N)
    
        
    def reset(self) :
        N = self.N
        #countess discarded and (prince or king) not discarded latter
        self.countess = deque([0]*N)
        #players have the same card
        #self.same = [[0 0 0 0],[0 0 0 0],[0 0 0 0],[0 0 0 0]]
        #soma de cada jogador
        self.tot_sum = deque([0]*N)
        #cartas já jogadas ( poderia colocar cartas restantes )
        self.cards_played = [5, 2 ,2 ,2 ,2 ,1, 1 ,1]
        #jogador já eliminado
        self.eliminated = deque([0]*N)     
        #imunidade de cada jogadaor
        self.imunity = deque([0]*N)
        #cartas na mão do jogador 0
        self.hand = [0 ,0] 
        #vetor de conhecimento público
        self.public = deque([[0, 0 ,0, 0 ,0 ,0 ,0 ,0]]*N) 
        #matriz de conhecimento privado (a linha x diz quem sabe de x )
        self.private = [[0, 0 ,0, 0],[0 ,0, 0, 0],[0 ,0 ,0, 0],[0, 0, 0 ,0]] #private knowledge of current player -> store inside player
        self.private = self.private[:N] + [[0, 0 ,0 ,0]]*(State.MAX_PLAYERS - N)
        
    def nextPlayer(self, player) :
        self.countess.rotate(-1)
        self.tot_sum.rotate(-1)
        self.eliminated.rotate(-1)
        self.victories.rotate(-1)
        self.imunity.rotate(-1)
        self.hand = list(sorted(player.hand))
        self.public.rotate(-1)
        self.private = player.private

    def stateToInput(self) :
        #FREQUENCY = [5, 2 ,2 ,2 ,2 ,1, 1 ,1]
        #don't forget to fill with 0's and transform deck into list
        #current player doesn't matter, as long as the shifts are made consistently
        public = []
        for pub in self.public :
            public += [x - 0.5 for x in pub]
        private = []
        for pri in self.private :
            private += [(x - 4)/8 for x in pri]
        cpl = [0]*(State.MAX_PLAYERS - self.N) #complement with 0's
        cpl1 = [1]*(State.MAX_PLAYERS - self.N) #complement with 1's
        cpl8 = [0]*( 8*State.MAX_PLAYERS - 8*self.N)
        
        WIN = (13 - 1)//self.N + 1 #number of rounds a player has to win 
        #79 parameters (they shoud be normalized between -0.5 and 0.5 )
        ret = [x - 0.5 for x in list(self.countess)] + cpl + [(self.N - State.MAX_PLAYERS/2)/State.MAX_PLAYERS]
        #print(len(ret))
        ret += [(x - State.MAX_SUM)/State.MAX_SUM for x in list(self.tot_sum)] + cpl
        #print(len(ret))
        ret += [(self.cards_played[i] - State.FREQUENCY[i]/2)/State.FREQUENCY[i] for i in range(len(self.cards_played))]
        #print(len(ret))
        ret += [x - 0.5 for x in list(self.eliminated)] + cpl1 + [(x - WIN/2)/WIN for x in list(self.victories)] + cpl
        #print(len(ret))
        ret += [x - 0.5 for x in list(self.imunity)] + cpl + [(x - 4)/8 for x in self.hand] +  public + cpl8 + private
        
        return ret
    

    def outputToPlay(self, output) :
        card = 0
        target = 0 
        guess = 0
        for i in range(len(output)) :
            if output[i] == 1 :
                ind = i + 1
                if ind < 33 :
                    card = 1
                    target = i//8 
                    guess = i%8 + 1
                elif ind >= 33 and ind <= 36 :
                    card = 2
                    target = ind - 33
                elif ind >= 37 and ind <= 40 :
                    card = 3
                    target = ind - 37
                elif ind == 41 :
                    card = 4
                elif ind >= 42 and ind <= 45 :
                    card = 5
                    target = ind - 42
                elif ind >= 46 and ind <= 49 :
                    card = 6
                    target = ind - 46
                elif ind == 50 :
                    card = 7
                elif ind == 51 :
                    card = 8
                break
        
        return card , target , guess

    def isValid(self , card , target , guess ) :
        N = self.N
        if target >= N or self.eliminated[target] :
            return False
        if not(card in self.hand) :
            
            return False
        if (card == 5 or card == 6) and (7 in self.hand) :
            return False
        if (card == 1 or card == 2 or card == 3 or card == 6) and target == 0 :
            return False
        if card == 4 and target != 0 :
            return False

        return True
               
