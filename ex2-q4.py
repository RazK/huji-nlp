import logging
import re
from collections import Counter
from itertools import product
from typing import List, Tuple, Dict, Any

from nltk.corpus import brown

from viterbi import calc_max_prob, EMISSIONS, TRANSITIONS, EMISSIONS_BIGRAM, TRANSITIONS_BIGRAM


def MLETagger(tagged_sentences: List[List[Tuple]]) -> Dict:
    """
    Compute the Maximum Likelihood Estimation for each word.
    :param tagged_sentences: list of sentences [[(word, tag), ...], ...]
    :return: dict {word : tag} where tag is maximizes p(tag|word)
    """
    word_counts = Counter()
    word_tag_counts = dict()

    for sentence in tagged_sentences:
        for (word, tag) in sentence:  # pair = (word, tag)
            if word not in word_tag_counts:
                word_tag_counts[word] = Counter()
            word_counts[word] += 1
            word_tag_counts[word][tag] += 1
    logging.debug(word_counts)
    logging.debug(word_tag_counts)

    mle_tag = lambda word: word_tag_counts[word].most_common(1)[0][0]
    word_tag_mle = {word: mle_tag(word) for word in word_tag_counts}
    logging.debug(word_tag_mle)

    return word_tag_mle


def KgramMLETagger(tagged_sentences: List[List[Tuple]], k=2, start_tag='START_TAG', end_tag='END_TAG') -> \
        Tuple[Dict[Tuple[str, str], float], \
              Dict[Tuple[Tuple, Tuple], float]]:
    """
    Calculate the emission and transition probabilities of the tags and words in the given tagged sentences.
    :param tagged_sentences: list of sentences [[(word, tag), ...], ...]
    :param k: 2 for Bigram, 3 for Trigram, etc.
    :param start_tag: A unique tag that signals the start of a sentence
    :param end_tag: A unique tag that signals the end of a sentence
    :return: (  emissions: {(tag, word) : probability},
                transitions: {((yk, yk-1, ..., y2), (yk-1, ..., y2, y1)) : probability} )
    """
    tag_counts = Counter()  # For each tag, counts the total number of words with this tag
    k_transitions = Counter()  # Counts the number of transitions (y | y-k, y-k+1, ..., y-1)
    k_1_transitions = Counter()  # Counts the number of transitions (y | y-k+1, ..., y-1)
    word_tag_counts = dict()  # Counts the number of occurrences of (word, tag) in the tagged sentences

    emissions = dict()  # e probabilities
    transitions = dict()  # q probabilities

    for sentence in tagged_sentences:
        # Make sure every sentence ends with a 'stop'
        sentence += [end_tag]

        # Begin every sentence with a 'start'
        kgram = [start_tag] * k  # [y-k, y-k+1, ..., y-2, y-1, y]
        for (word, tag) in sentence:
            # Accumulate the number of occurrences of each (word,tag) pair
            if word not in word_tag_counts.keys():
                word_tag_counts[word] = Counter()
            word_tag_counts[word][tag] += 1

            # Accumulate the number of occurrences of each tag
            tag_counts[tag] += 1

            # Accumulate k-gram and k-1-gram transitions
            # k-gram:   [y-k, y-k+1, ..., y-2, y-1, y]
            # k-1-gram: [y-k, y-k+1, ..., y-2, y-1]
            k_transitions[tuple(kgram)] += 1
            k_1_transitions[tuple(kgram[:-1])] += 1
            kgram.pop(0)  # remove [y-k]
            kgram.append(tag)  # add new [y]

    #             word_tag_counts[x1][y1]
    #  e(x1|y1) = -----------------------
    #                  tag_counts[y1]
    for sentence in tagged_sentences:
        for (word, tag) in sentence:
            emissions[(word, tag)] = word_tag_counts[word][tag] / tag_counts[tag]

    #                        k_transitions[(y-k, y-k+1, ..., y-1, y)]
    #  q(y|y-k, ..., y-1) =  ----------------------------------------
    #                           k_1_transitions[(y-k, ..., y-1)]
    for k_gram in k_transitions:
        prev_gram = k_gram[:-1]  # (y-k, y-k+1, ..., y-1)
        cur_gram = k_gram[1:]  # (y-k+1, ..., y-1, y)
        transitions[(prev_gram, cur_gram)] = k_transitions[k_gram] / k_1_transitions[prev_gram]

    return emissions, transitions


def getErrorRate(train_data, test_data, MLETagger, default_tag='NN'):
    """
    Calculate the error rate (1-accuracy) of the MLE Tagger.
    :param train_data: list of tagged sentences [[(word, tag), ...], ...]
    :param test_data:  list of tagged sentences [[(word, tag), ...], ...]
    :param MLETagger:  function(tagged_sentences) -->
    :return: tuple (1-accuracyTotal, 1-accuracyKnown, 1-accuracyUnknown)
    """
    train_mle = MLETagger(train_data)
    predictions = dict()
    totalWords = {'known': 0,
                  'unknown': 0}
    correctWords = {'known': 0,
                    'unknown': 0}

    flatTestData = [(word, tag) for sentence in test_data
                    for (word, tag) in sentence]

    # Go over all test data and count correct predictions
    for (word, tag) in flatTestData:
        # Known words
        if word in train_mle:
            predictions[word] = train_mle[word]
            type = 'known'

        # Unknown words
        else:
            predictions[word] = default_tag
            type = 'unknown'

        totalWords[type] += 1
        if tag == predictions[word]:
            correctWords[type] += 1

    accuracyKnown = correctWords['known'] / totalWords['known']
    accuracyUnknown = correctWords['unknown'] / totalWords['unknown']
    accuracyTotal = \
        (correctWords['known'] + correctWords['unknown']) / len(flatTestData)

    return (accuracyTotal, accuracyKnown, accuracyUnknown)


def get_viterbi_tagging(untagged_sentence, emissions, transitions, tags, k=2, start_word='START_WORD',
                        end_word='END_WORD', start_tag='START_TAG', end_tag='END_TAG'):
    """
    Calculate the most likely sentence tagging given the emissions and transitions.
    :param untagged_sentence: list of words [word, word, ...]
    :param emissions: dict {(tag, word) : probability}
    :param transitions: dict {((tag_k, tag_k-1, ... tag_2), (tag_k-1, ..., tag_2, tag_1)) : probability}
    :param tags: set of all possible tags
    :param k: 2 for Bigram, 3 for Trigram, etc.
    :param start_tag: A unique character that never appears in any word, used as placeholder for starting words
    :return: tagged_sentence List[(word, tag), ...]
    """
    # Extend the sentence with (k-1) start tokens to slide over it with a (k-1)-window
    # For example: sentence = 'ABCD', k=3, start_word='*', stop_word='$'
    #              padded_sentence = '**ABCD$'
    padded_sentence = (k - 1) * [start_word] + untagged_sentence + [end_word]

    # Calculate all (k-1)-words in the sliding (k-1)-window over the input sentence
    # For example: sentence = 'ABCD', k=3, start_word='*', stop_word='$'
    #              padded_sentence = '**ABCD'
    #              k_1_word = '**', '*A', 'AB', 'BC', 'CD', 'D$'
    all_k_1_words = [tuple(padded_sentence[i:i + k-1]) for i in range(len(sentence) + 1)]

    # Calculate all possible (k-1)-tags combinations
    # For example: tags = ['y1', 'y2', 'y3'], k=2
    #              all_k_1_tags = [['y1', 'y1'], ['y1', 'y2'], ['y1', 'y3'],
    #                              ['y2', 'y1'], ['y2', 'y2'], ['y2', 'y3'],
    #                              ['y3', 'y1'], ['y3', 'y2'], ['y3', 'y3']]
    padded_tags = tags | {start_tag, end_tag}
    all_k_1_tags = set(product(padded_tags, repeat=k - 1))  # k-1 because size(Bigram tags)=1, size(Trigram tags)=2, etc.
    all_k_tags = set(product(padded_tags, repeat=k))

    # Initialize all (k_1_words, k_1_tags) probabilities
    # [{(k-1)-tag : p,
    #   (k-1)-tag : p,
    #                ...}
    # ...
    # ]
    max_k_1_words_tags_probabilities = [{tuple(k_1_tags): 0 for k_1_tags in all_k_1_tags} for k_1_words in all_k_1_words]

    # Set all first layer probabilities to 1
    for k_1_tags in max_k_1_words_tags_probabilities[0]:
        # skip the end-tag
        if k_1_tags == end_tag:
            continue
        max_k_1_words_tags_probabilities[0][k_1_tags] = (1, (0,start_tag))

    # Viterbi over all following probabilities
    for i in range(1, len(all_k_1_words)):
        prev_k_1_words = all_k_1_words[i - 1]
        cur_k_1_words = all_k_1_words[i]
        for k_tags in all_k_tags:
            prev_k_1_tags = k_tags[:-1]  # (y-k, ..., y-1)
            cur_k_1_tags = k_tags[1:]  # (y-k+1, ..., y)
            transit = (prev_k_1_tags, cur_k_1_tags)  # ((y-k, ..., y-1), (y-k+1, ..., y))
            emit = (cur_k_1_tags[-1], cur_k_1_words[-1])  # (y, x)
            transit_prob = emit_prob = 0
            prev_prob = max_k_1_words_tags_probabilities[i-1][prev_k_1_tags][0]
            # TODO: SB - emit_prob = emissions((k-1)*'NN', )
            if transit in transitions:
                # set to transit
                transit_prob = transitions[transit]
            if emit in emissions:
                emit_prob = emissions[emit]
            prob = prev_prob * transit_prob * emit_prob

            # Update max-probabilities if this is the best option so far
            if prob > max_k_1_words_tags_probabilities[i][tuple(cur_k_1_tags)]:
                max_k_1_words_tags_probabilities[i][tuple(cur_k_1_tags)] = (prob, (i-1, prev_k_1_tags))   # TODO: (prob, back_pointer) .... back_pointer = (i-1,prev_k_1_tags)

    # Extract max probability tags and return
    best_end = max(max_k_1_words_tags_probabilities[-1], key=lambda tags:max_k_1_words_tags_probabilities[-1].get(tags)[0])

    tags = [max(tags_layer, key=lambda tags:tags_layer.get(tags)[0]) for tags_layer in max_k_1_words_tags_probabilities]
    return [(sentence[i], tags[i]) for i in range(len(sentence))]


def simplifyTags(tagged_sentences: List[List[Tuple]]) -> List[List[Tuple]]:
    """
    Strip '-' and '+' signs (and whatever comes afterwards ) from the tags.
    For example:
        NN-TL   --> NN
        NR-HL   --> NR
        NP+BEZ  --> NP
    Return the same sentences list with simplified tags.
    :param tagged_sentences: list of sentences [[(word, tag), ...], ...]
    :return: same sentences list with simplified tags
    """
    simplify = lambda tag: re.split('[\+,\-]', tag)[0]
    return [[(word, simplify(tag)) for (word, tag) in sentence]
            for sentence in tagged_sentences]


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # http://www.nltk.org/book/ch05.html
    brown_tagged_sents = brown.tagged_sents(categories='news')
    brown_simplified_tagged_sents = simplifyTags(brown_tagged_sents)
    brown_sents = brown.sents(categories='news')
    size = int(len(brown_simplified_tagged_sents) * 0.9)
    train_sents = brown_simplified_tagged_sents[:size]
    test_sents = brown_simplified_tagged_sents[size:]

    # word_probabilities = MLETagger(train_sents)
    # logging.info(word_probabilities)

    # accuracies = getErrorRate(train_sents, test_sents, MLETagger)
    # logging.info(accuracies)

    # train_emissions, train_transitions = KgramMLETagger(train_sents)
    train_emissions = EMISSIONS_BIGRAM
    train_transitions = TRANSITIONS_BIGRAM
    train_sents = [['A', 'C', 'C', 'G', 'T', 'G', 'C', 'A']]
    logging.info('Emissions: {}'.format(train_emissions))
    logging.info('Transitions: {}'.format(train_transitions))

    for sentence in train_sents:
        print(get_viterbi_tagging(sentence, train_emissions, train_transitions, tags={'H', 'L'}))
