TRANSITIONS = {'H' : {'H' : 0.5,
                      'L' : 0.5},
               'L' : {'H' : 0.4,
                      'L' : 0.6}}

EMISSIONS = {'H' : {'A' : 0.2,
                    'C' : 0.3,
                    'G' : 0.3,
                    'T' : 0.2},
             'L' : {'A' : 0.3,
                    'C' : 0.2,
                    'G' : 0.2,
                    'T' : 0.3}}

STATES = TRANSITIONS.keys()

def calc_max_prob(sequence, prev_state):
    # Init state max probabilities table
    #   0   1   2   ...     n
    #   H   H   H   ...     H
    #   L   L   L   ...     L
    init_state_probabilities = [{state : 1 if state == prev_state else 0
                                 for state in STATES}]
    next_state_probabilities = [{state : 0 for state in STATES}
                                for i in range(len(sequence))]
    max_state_probabilities = init_state_probabilities + \
                              next_state_probabilities
    # Calculate max probabilities
    # character : {'A', 'C', 'G', 'T'}
    for i, character in enumerate(' '+sequence):
        # Skip init state
        if i == 0:
            continue
        # Update max_state_probabilities table
        for prev_state in STATES:  # state : {'H', 'L'}
            for cur_state in STATES:
                prob =  max_state_probabilities[i-1][prev_state] * \
                        TRANSITIONS[prev_state][cur_state] * \
                        EMISSIONS[cur_state][character]
                if prob > max_state_probabilities[i][cur_state]:
                    max_state_probabilities[i][cur_state] = prob

    # Print max probability sequence
    for layer in max_state_probabilities:
        print(max(layer, key=layer.get), end=" ")
        print(layer)

if __name__ == '__main__':
    calc_max_prob("ACCGTGCA", 'H')