network = Network()
network.loadParameters('weights_4.picke')



#inside game dynamics
( state)
    inp = stateToInput(state)
    logit = network.getLogits(inp)
    card , target , guess , encoding = player.choose_play(logit , state)
    print card , target ,guess
    return card , target , guess
#from raw_input
play = card , target , guess , outcome

#pega o outcome e transforma numa jogada
(card , target , guess , outcome ) #from raw_input)
     play
    
     
    (play , state)

new state


(play , state)
new state


