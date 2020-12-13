TRANSITIONS = {('H', 'H'): 0.5,
               ('H', 'L'): 0.5,
               ('L', 'H'): 0.4,
               ('L', 'L'): 0.6}

EMISSIONS = {('H', 'A'): 0.2,
             ('H', 'C'): 0.3,
             ('H', 'G'): 0.3,
             ('H', 'T'): 0.2,
             ('L', 'A'): 0.3,
             ('L', 'C'): 0.2,
             ('L', 'G'): 0.2,
             ('L', 'T'): 0.3}

TAGS = {'H', 'L'}

TRANSITIONS_BIGRAM = {(('START_TAG',), ('H',)): 0.5,
                      (('START_TAG',), ('L',)): 0.5,
                      (('H',), ('H',)): 0.5,
                      (('H',), ('L',)): 0.5,
                      (('L',), ('H',)): 0.4,
                      (('L',), ('L',)): 0.6,
                      (('L',), ('END_TAG',)): 0.5,
                      (('H',), ('END_TAG',)): 0.5}

EMISSIONS_BIGRAM = {('H', 'A'): 0.2,
                    ('H', 'C'): 0.3,
                    ('H', 'G'): 0.3,
                    ('H', 'T'): 0.2,
                    ('L', 'A'): 0.3,
                    ('L', 'C'): 0.2,
                    ('L', 'G'): 0.2,
                    ('L', 'T'): 0.3,
                    ('END_TAG', 'END_WORD'): 1}

TAGS_BIGRAM = {('H',), ('L',)}


TRANSITIONS_DOGCAT = {(('START_TAG',), ('dog',)): 1,
                      (('START_TAG',), ('cat',)): 0,
                      (('cat',), ('cat',)): 0.5,
                      (('cat',), ('dog',)): 0,
                      (('dog',), ('cat',)): 0.25,
                      (('dog',), ('dog',)): 0.5,
                      (('cat',), ('END_TAG',)): 0.5,
                      (('dog',), ('END_TAG',)): 0.25}

EMISSIONS_DOGCAT = {('cat', 'woof'): 0.5,
                    ('cat', 'meow'): 0.5,
                    ('dog', 'woof'): 0.75,
                    ('dog', 'meow'): 0.25,
                    ('END_TAG', 'END_WORD'): 1}

def calc_max_prob(sequence, prev_state, emissions, transitions, tags):
    # Init state max probabilities table
    #   0   1   2   ...     n
    #   H   H   H   ...     H
    #   L   L   L   ...     L
    init_state_probabilities = [{state: 1 if state == prev_state else 0
                                 for state in tags}]
    next_state_probabilities = [{state: 0 for state in tags}
                                for i in range(len(sequence))]
    max_state_probabilities = init_state_probabilities + \
                              next_state_probabilities
    # Calculate max probabilities
    # character : {'A', 'C', 'G', 'T'}
    for i, character in enumerate(' ' + sequence):
        # Skip init state
        if i == 0:
            continue
        # Update max_state_probabilities table
        for prev_state in tags:  # state : {'H', 'L'}
            for cur_state in tags:
                transit = (prev_state, cur_state)
                emit = (cur_state, character)
                prob = max_state_probabilities[i - 1][prev_state] * \
                       transitions[transit] * \
                       emissions[emit]

                if prob > max_state_probabilities[i][cur_state]:
                    max_state_probabilities[i][cur_state] = prob

    # Print max probability sequence
    for layer in max_state_probabilities:
        print(max(layer, key=layer.get), end=" ")
        print(layer)


if __name__ == '__main__':
    calc_max_prob("ACCGTGCA", 'H', EMISSIONS, TRANSITIONS, TAGS)
