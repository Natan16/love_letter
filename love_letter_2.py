from state import State
print "type the number of players"
N = raw_input()

print "type the first player to play ( you are player 1 )"
i = raw_input()


print "type your hand"
raw_input()
s = State(N)



    if i != 1 :
        print "type player " + i + " card ,target and guess and outcome"
        card = raw_input()
        target = raw_input()
        guess = raw_input()
        outcome = raw_input() #translate outcome to state

    else :
        print "type the card you drew"
        raw_input()
        
        card = 0
        print "you should play the card " + card
    
    
