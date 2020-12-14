import logging
import re
from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Dict
from nltk.corpus import brown

START_TAG = 'START_TAG'
END_TAG = 'END_TAG'
START_WORD = 'START_WORD'
END_WORD = 'END_WORD'

PSEUDO_REGEX = \
    {'_MONEY': '.*\$.*',  # https://regex101.com/r/UoFL0T/1
     '_NUMBER': '\d+',  # https://regex101.com/r/kzCwuJ/2
     '_NUMBERED': '\d+(st|nd|rd|th)',  # https://regex101.com/r/0rLKtF/1
     '_FUL': '.*ful',
     '_EST': '.*est',
     '_ER': '.*er',
     '_TION': '.*tion',
     '_%': '.*%.*',
     '_.': '.*\..*',  # https://regex101.com/r/Q8H1v1/1
     '_-': '.*-.*'  # https://regex101.com/r/Q8H1v1/2
     }
PSEUDO_REGEX_COMPILED = {tag: re.compile(PSEUDO_REGEX[tag]) for tag in
                         PSEUDO_REGEX}


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


def getErrorRateBigram(train_data, test_data, train_predictions,
                       default_tag='NN'):
    """
    Calculate the error rate (1-accuracy) of the MLE Tagger.
    :param train_data: list of tagged sentences [[(word, tag), ...], ...]
    :param test_data:  list of tagged sentences [[(word, tag), ...], ...]
    :param MLETagger:  function(tagged_sentences) -->
    :return: tuple (1-accuracyTotal, 1-accuracyKnown, 1-accuracyUnknown)
    """
    train_mle = MLE_tagger(train_data)
    totalWords = {'known': 0,
                  'unknown': 0}
    correctWords = {'known': 0,
                    'unknown': 0}

    flatTestData = [(word, tag) for sentence in test_data
                    for (word, tag) in sentence]

    # Go over all test data and count correct predictions
    for i, (word, tag) in enumerate(flatTestData):
        # Known words
        if word in train_mle:
            type = 'known'

        # Unknown words
        else:
            type = 'unknown'

        totalWords[type] += 1
        if tag == train_predictions[i][1]:
            correctWords[type] += 1

    accuracyKnown = correctWords['known'] / totalWords['known']
    accuracyUnknown = correctWords['unknown'] / totalWords['unknown']
    accuracyTotal = \
        (correctWords['known'] + correctWords['unknown']) / len(flatTestData)

    return (accuracyTotal, accuracyKnown, accuracyUnknown)


def MLE_tagger(tagged_sentences: List[List[Tuple]]) -> Dict:
    """
    Compute the Maximum Likelihood Estimation for each word.
    :param tagged_sentences: list of sentences [[(word, tag), ...], ...]
    :return: dict {word : tag} where tag maximizes p(tag|word)
    """
    word_counts = Counter()
    word_tag_counts = dict()

    for sentence in tagged_sentences:
        for (word, tag) in sentence:
            if word not in word_tag_counts:
                word_tag_counts[word] = Counter()
            word_counts[word] += 1
            word_tag_counts[word][tag] += 1

    mle_tag = lambda word: word_tag_counts[word].most_common(1)[0][0]
    word_tag_mle = {word: mle_tag(word) for word in word_tag_counts}

    return word_tag_mle


def bigram_HMM(tagged_sentences: List[List[Tuple]],
               add_one_smoothing: bool = True) -> \
        Tuple[Dict[Tuple[str, str], float], \
              Dict[Tuple[Tuple, Tuple], float]]:
    """
    Calculate the emission and transition probabilities of the tags and words
    in the given tagged sentences.
    :param tagged_sentences: list of sentences [[(x, y), ...], ...]
    :return: (  emissions: {(y, x) : probability},
                transitions: {((y-1), (y)) : probability} )
    """
    # For each tag, counts the total number of words with this tag
    tag_counts = Counter()

    # For each word, for each tag, counts the number of occurrences of
    # (word, tag) in the tagged sentences
    word_tag_counts: Dict[Counter] = dict()

    # Counts the number of (y-1, y) in the tagged sentences
    bigrams = Counter()

    # Counts the number of (y) in the tagged sentences
    unigrams = Counter()

    # e probabilities
    emissions = dict()

    # q probabilities
    transitions = dict()

    # Wrap every sentence with 'start' and 'stop' symbols
    padded_tagged_sentences = \
        [[(START_WORD, START_TAG)] + sentence + [(END_WORD, END_TAG)] for
         sentence in tagged_sentences]
    for sentence in padded_tagged_sentences:

        # Sliding bigram window over the sentence
        tags_bigram = [sentence[0][1], ]  # start tag

        for (word, tag) in sentence[1:]:
            tags_bigram.append(tag)  # add new [y]

            # Accumulate the number of occurrences of each tag
            tag_counts[tag] += 1

            # Accumulate the number of occurrences of each (word,tag) pair
            if word not in word_tag_counts.keys():
                word_tag_counts[word] = Counter()
            word_tag_counts[word][tag] += 1

            # Accumulate bigram and unigram occurrences
            # bigram:  [y-1, y]
            # unigram: [y-1]
            bigrams[tuple(tags_bigram)] += 1
            unigrams[tuple(tags_bigram[:-1])] += 1

            tags_bigram.pop(0)  # remove [y-1]

    #           word_tag_counts[x][y]
    #  e(x|y) = ---------------------
    #               tag_counts[y]
    tag_word_counts = {tag: {word: word_tag_counts[word][tag] for word in
                             word_tag_counts if tag in word_tag_counts[word]}
                       for tag in tag_counts}
    for sentence in padded_tagged_sentences:
        for (word, tag) in sentence[1:]:  # skip 'start' tag
            if add_one_smoothing:
                emissions[(tag, word)] = \
                    (word_tag_counts[word][tag] + 1) / \
                    (tag_counts[tag] + len(tag_word_counts[tag]))
            else:
                emissions[(tag, word)] = \
                    word_tag_counts[word][tag] / \
                    tag_counts[tag]

    emissions[(END_TAG, END_WORD)] = 1

    #              bigrams[(y-1, y)]
    #  q(y|y-1) =  -----------------
    #               unigrams[(y-1)]
    for tags_bigram in bigrams:  # (y-1, y)
        prev_gram = tags_bigram[:-1]  # (y-1)
        cur_gram = tags_bigram[1:]  # (y)
        transitions[(prev_gram, cur_gram)] = \
            bigrams[tags_bigram] / unigrams[prev_gram]

    return emissions, transitions


def bigram_viterbi(untagged_sentence,
                   emissions,
                   transitions,
                   tags,
                   unknown_tag='NN'):
    """
    Calculate the most likely sentence tagging given the emissions and
    transitions.
    :param untagged_sentence: list of words [x1, x2, ...]
    :param emissions: {(y, x) : probability}
    :param transitions: {((y-1), (y)) : probability} )
    :param tags: set of all possible tags
    :param unknown_tag: A tag used for unknown words
    :return: tagged_sentence List[(x1, y1), (x2, y2), ...]
    """
    padded_sentence = [START_WORD, ] + untagged_sentence + [END_WORD, ]
    padded_tags = tags | {START_TAG, END_TAG}
    viterbi = {i: {tag: 0 for tag in padded_tags} for i in
               range(len(padded_sentence))}
    backpointers = {i: {tag: None for tag in padded_tags} for i in
                    range(len(padded_sentence))}
    viterbi[0][START_TAG] = 1

    # Set all unknown words (no emission) to have the tag 'NN'
    for word in untagged_sentence:
        has_emissions = False
        for cur_tag in padded_tags:
            emit = (cur_tag, word)
            if emit in emissions:
                has_emissions = True
        if (not has_emissions):
            emissions[('NN', word)] = 1

    for i in range(1, len(padded_sentence)):
        word = padded_sentence[i]

        for prev_tag in padded_tags:
            for cur_tag in padded_tags:
                transit = ((prev_tag,), (cur_tag,))
                emit = (cur_tag, word)
                if transit not in transitions:
                    transition = 1 / (10 ** 30)  # TODO  RK - explain why
                else:
                    transition = transitions[transit]

                if emit not in emissions:
                    emission = 0
                else:
                    emission = emissions[emit]

                prev_prob = viterbi[i - 1][prev_tag]
                cur_prob = prev_prob * \
                           transition * \
                           emission

                if (cur_prob > viterbi[i][cur_tag]):
                    viterbi[i][cur_tag] = cur_prob
                    backpointers[i][cur_tag] = prev_tag

    best_tag_sequence = []
    i = len(padded_sentence) - 1
    best_tag = backpointers[i][END_TAG]
    while best_tag != START_TAG:  # while i >= 0:
        i -= 1
        best_tag_sequence.append(best_tag)
        best_tag = backpointers[i][best_tag]
    best_tag_sequence.reverse()

    return [(untagged_sentence[i], best_tag_sequence[i])
            for i in range(len(untagged_sentence))]


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


def get_low_frequency_words(all_tagged_sents, train_tagged_sents):
    """
    Return a list of words from the dataset that have a low frequency in the
    training sentences.
    :param all_tagged_sents: list [[(word,tag), ...], ...]
    :param train_tagged_sents: list [[(word, tag), ...], ...]
    :return: dict {word : frequency in training} for all words in dataset if
    their frequency in training is less than 5
    """
    # Split vocabulary to two parts:
    # 1) appear at least  5 times in training (high freq)
    # 2) appear less than 5 times in training (low freq)
    train_word_counts = Counter()
    for sentence in train_tagged_sents:
        for (word, tag) in sentence:
            train_word_counts[word] += 1

    low_freq = {word: train_word_counts[word]
                for sentence in all_tagged_sents
                for (word, tag) in sentence
                if train_word_counts[word] < 5}

    return low_freq


def get_pseudo_class(word):
    """
    Returns the first pseudo-tag that matches this word
    :param word: a word from the dataset
    :return: a tag from PSEUDO_REGEX that matches it
    """
    for tag in PSEUDO_REGEX_COMPILED:
        if re.match(PSEUDO_REGEX_COMPILED[tag], word):
            return tag


def apply_pseudo_classes(all_sents, train_sents, test_sents):
    """
    Replace tags of all low-freq words in train and test sents with pseudo
    classes, and return new lists of pseudo-tagged sentences.
    :param all_sents: list [[(word,tag), ...], ...]
    :param train_sents: list [[(word,tag), ...], ...]
    :param test_sents: list [[(word,tag), ...], ...]
    :return: (train_sents_pseudo, test_sents_pseudo)
    """
    train_sents_pseudo = deepcopy(train_sents)
    test_sents_pseudo = deepcopy(test_sents)

    low_freq_words = get_low_frequency_words(all_sents, train_sents)

    # Pseudo-ify the train sentences
    for sentence in train_sents_pseudo:
        for j, (word, tag) in enumerate(sentence):
            if word in low_freq_words:
                pseudo_tag = get_pseudo_class(word)
                sentence[j] = (word, pseudo_tag)

    # Pseudo-ify the test sentences
    for sentence in test_sents_pseudo:
        for j, (word, tag) in enumerate(sentence):
            if word in low_freq_words:
                pseudo_tag = get_pseudo_class(word)
                sentence[j] = (word, pseudo_tag)

    return (train_sents_pseudo, test_sents_pseudo)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # http://www.nltk.org/book/ch05.html
    brown_tagged_sents = brown.tagged_sents(categories='news')
    brown_simplified_tagged_sents = simplifyTags(brown_tagged_sents)
    simplified_tags = {tag for sentence in brown_simplified_tagged_sents
                       for (word, tag) in sentence}
    size = int(len(brown_simplified_tagged_sents) * 0.9)
    train_sents = brown_simplified_tagged_sents[:size]
    test_sents = brown_simplified_tagged_sents[size:]
    test_sents_untagged = [[word for (word, tag) in sentence]
                           for sentence in test_sents]

    # MLE tagger
    accuracies = getErrorRate(train_sents, test_sents, MLE_tagger)
    logging.info(accuracies)

    # Bigram HMM + Viterbi tagger
    train_emissions, train_transitions = \
        bigram_HMM(train_sents, add_one_smoothing=False)
    predictions = []
    for untagged_sentence in test_sents_untagged:
        new_lst = bigram_viterbi(untagged_sentence,
                                 train_emissions,
                                 train_transitions,
                                 tags=simplified_tags)
        predictions = predictions + new_lst
    accuracies = getErrorRateBigram(train_sents, test_sents, predictions)
    logging.info(accuracies)

    # Bigram HMM + Add-1 Smoothing + Viterbi tagger
    train_emissions_smoothed, train_transitions_smoothed = \
        bigram_HMM(train_sents, add_one_smoothing=True)
    predictions_smoothed = []
    for untagged_sentence in test_sents_untagged:
        new_lst = bigram_viterbi(untagged_sentence,
                                 train_emissions_smoothed,
                                 train_transitions_smoothed,
                                 tags=simplified_tags)
        predictions_smoothed = predictions_smoothed + new_lst
    accuracies = getErrorRateBigram(train_sents, test_sents,
                                    predictions_smoothed)
    logging.info(accuracies)

    # Bigram HMM + Pseudo tags + Viterbi tagger
    train_sents_pseudo, test_sents_pseudo = \
        apply_pseudo_classes(brown_simplified_tagged_sents,
                             train_sents,
                             test_sents)
    train_emissions_pseudo, train_transitions_pseudo = \
        bigram_HMM(train_sents_pseudo, add_one_smoothing=False)
    predictions_pseudo = []
    for untagged_sentence in test_sents_untagged:
        new_lst = bigram_viterbi(untagged_sentence,
                                 train_emissions_pseudo,
                                 train_transitions_pseudo,
                                 tags=simplified_tags)
        predictions_pseudo = predictions_pseudo + new_lst
    accuracies = getErrorRateBigram(train_sents_pseudo,
                                    test_sents_pseudo,
                                    predictions_pseudo)
    logging.info(accuracies)
