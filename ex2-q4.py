import logging
import re
from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Dict
from nltk.corpus import brown
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

START_TAG = 'START_TAG'
END_TAG = 'END_TAG'
START_WORD = 'START_WORD'
END_WORD = 'END_WORD'

PSEUDO_REGEX = \
    {'_MONEY': '.*\$.*',  # https://regex101.com/r/UoFL0T/1
     '_NUMBER': '\d+',  # https://regex101.com/r/kzCwuJ/2
     '_NUMBERED': '\d+(st|nd|rd|th)',  # https://regex101.com/r/0rLKtF/1
     '_%': '.*%.*',
     '_.': '.*\..*',  # https://regex101.com/r/Q8H1v1/1
     '_-': '.*-.*',  # https://regex101.com/r/Q8H1v1/2
     '_FUL': '.*ful$',
     '_EST': '.*est$',
     '_ER': '.*er$',
     '_TION': '.*tion$',
     '_\'S': '.*\'s?'
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

    return (1 - accuracyTotal, 1 - accuracyKnown, 1 - accuracyUnknown)


def getErrorRateBigram(train_data, test_data, predictions, all_tags,
                       plot_confusion_matrix=False,
                       default_tag='NN'):
    """
    Calculate the error rate (1-accuracy) of the MLE Tagger.
    :param train_data: list of tagged sentences [[(word, tag), ...], ...]
    :param test_data:  list of tagged sentences [[(word, tag), ...], ...]
    :param all_tags: set of all tags in the data (train + test)
    :param predictions: list of tagged words [(word, tag), ...]
    :param plot_confusion_matrix: set to True to plot.
    :return: tuple (1-accuracyTotal, 1-accuracyKnown, 1-accuracyUnknown)
    """
    train_mle = MLE_tagger(train_data)
    total_words = {'known': 0,
                  'unknown': 0}
    correct_words = {'known': 0,
                    'unknown': 0}

    flat_test_data = [(word, tag) for sentence in test_data
                    for (word, tag) in sentence]

    if plot_confusion_matrix:
        test_true_tags = [tag for sentence in test_data
                          for (word, tag) in sentence]
        test_pred_tags = [tag for (word, tag) in predictions]
        plot_confusion_matrix(test_true_tags, test_pred_tags, all_tags)

    # Go over all test data and count correct predictions
    for i, (word, tag) in enumerate(flat_test_data):
        # Known words
        if word in train_mle:
            type = 'known'

        # Unknown words
        else:
            type = 'unknown'

        total_words[type] += 1
        if tag == predictions[i][1]:
            correct_words[type] += 1

    accuracyKnown = correct_words['known'] / total_words['known']
    accuracyUnknown = correct_words['unknown'] / total_words['unknown']
    accuracyTotal = \
        (correct_words['known'] + correct_words['unknown']) / len(flat_test_data)

    return (1 - accuracyTotal, 1 - accuracyKnown, 1 - accuracyUnknown)


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
    total_counts = sum(tag_counts.values())
    for sentence in padded_tagged_sentences:
        for (word, tag) in sentence[1:]:  # skip 'start' tag
            if add_one_smoothing:
                emissions[(tag, word)] = \
                    (word_tag_counts[word][tag] + 1) / \
                    (tag_counts[tag] + total_counts)
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

    return emissions, transitions, tag_counts


def bigram_viterbi(untagged_sentence,
                   emissions,
                   transitions,
                   tags,
                   tag_counts,
                   add_one_smoothing,
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
    total_counts = sum(tag_counts.values())

    # Set all unknown words (no emission) to have the tag 'NN'
    for word in untagged_sentence:
        has_emissions = False
        for cur_tag in padded_tags:
            emit = (cur_tag, word)
            if emit in emissions:
                has_emissions = True
        if (not has_emissions):
            if not add_one_smoothing:
                emissions[(unknown_tag, word)] = 1/tag_counts[unknown_tag]

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
                    if add_one_smoothing:
                        emission = 1.0 / (total_counts + tag_counts[cur_tag])
                    else:
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
                # Attempt adding word to a pseudo tag
                pseudo_tag = get_pseudo_class(word)
                if pseudo_tag:
                    sentence[j] = (word, pseudo_tag)
                # Otherwise use the original tag
                else:
                    sentence[j] = (word, tag)

    # Pseudo-ify the test sentences
    for sentence in test_sents_pseudo:
        for j, (word, tag) in enumerate(sentence):
            if word in low_freq_words:
                # Attempt adding word to a pseudo tag
                pseudo_tag = get_pseudo_class(word)
                if pseudo_tag:
                    sentence[j] = (word, pseudo_tag)
                # Otherwise use the original tag
                else:
                    sentence[j] = (word, tag)

    return (train_sents_pseudo, test_sents_pseudo)


def mle_tagger(tagged_train_data,
               tagged_test_data):
    """
    Train a Maximum Likelihood Estimator on the tagged train data, then try
    it on the test data and report the resulting accuracies.
    :param tagged_train_data: list [[(word, tag), ...], ...]
    :param tagged_test_data: list [[(word, tag), ...], ...]
    :return: tuple (1-accuracyTotal, 1-accuracyKnown, 1-accuracyUnknown)
    """
    accuracies = getErrorRate(tagged_train_data, tagged_test_data, MLE_tagger)
    logging.info(accuracies)
    return accuracies


def bigram_hmm_viterbi(tagged_all_data,
                       tagged_train_data,
                       tagged_test_data,
                       untagged_test_data,
                       all_tags,
                       smoothing=False,
                       pseudo_classes=False,
                       plot_confusion_matrix=False):
    """
    Train a Bigram Hidden-Markov-Model on the train data and acquire emission
    and transition probabilities.
    optional:
        * Use 'Add-1' smoothing on the emission probabilities
        * Use pseudo classes to cluster words with low frequency
    Run the Viterbi sequence tagging algorithm on the test data and report the
    resulting accuracies.
    :param tagged_train_data: list [[(word, tag), ...], ...]
    :param tagged_test_data: list [[(word, tag), ...], ...]
    :param untagged_test_data: list [[word, ...], ...]
    :param all_tags: {tag1, tag2, ...}
    :param smoothing: bool - apply Add-1 Smoothing to emission probabilities?
    :param pseudo_classes: bool - cluster low frequency words in
                                  pseudo-classes?
    :param plot_confusion_matrix: bool - True to plot confusion matrix.
    :return: tuple (1-accuracyTotal, 1-accuracyKnown, 1-accuracyUnknown)
    """
    # Refactor pseudo-classes for low frequency words if required
    if (pseudo_classes):
        tagged_train_data, tagged_test_data = \
            apply_pseudo_classes(tagged_all_data,
                                 tagged_train_data,
                                 tagged_test_data)
        all_tags |= {tag for tag in PSEUDO_REGEX}

    # Train a Bigram HMM and get emissions and transitions
    train_emissions, train_transitions, tag_counts = \
        bigram_HMM(tagged_train_data, add_one_smoothing=smoothing)

    # Run Viterbi on test data and return accuracies of the results
    predictions = []
    for untagged_sentence in untagged_test_data:
        new_lst = bigram_viterbi(untagged_sentence,
                                 train_emissions,
                                 train_transitions,
                                 all_tags,
                                 tag_counts,
                                 add_one_smoothing=smoothing)
        predictions = predictions + new_lst
    accuracies = getErrorRateBigram(tagged_train_data,
                                    tagged_test_data,
                                    predictions,
                                    all_tags,
                                    plot_confusion_matrix=plot_confusion_matrix)
    logging.info(accuracies)
    return accuracies


def plot_confusion_matrix(y_true=["cat", "ant", "cat", "cat", "ant", "bird"],
                          y_pred=["ant", "ant", "cat", "cat", "ant", "cat"],
                          tags=["ant", "bird", "cat"]):
    # sphinx_gallery_thumbnail_number = 2

    harvest = confusion_matrix(y_true, y_pred, labels=tags)

    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(tags)))
    ax.set_yticks(np.arange(len(tags)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(tags)
    ax.set_yticklabels(tags)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(tags)):
        for j in range(len(tags)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix (True / Predicted)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fig.tight_layout()
    plt.show()

def main():
    plot_confusion_matrix(y_true=["NN","CYBER","NX","GAY","CYBER","NN"],
                          y_pred=["NN","NN","CYBER","NN","CYBER","NN"],
                          tags=["CYBER","NN","GAY","NX","MOTEK"])

    """
    Load the Brown corpus and perform several word tagging learning
    experiments, including MLE (Maximum Likelihood Estimation), Bigram HMM +
    Viterbi, Smoothing, Pseudo classes.
    """
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

    # MLE Tagger
    logging.info("MLE Tagger")
    mle_tagger(train_sents, test_sents)

    # Bigram HMM -> Viterbi Sequence Tagger
    logging.info("Bigram HMM --> Viterbi")
    bigram_hmm_viterbi(brown_simplified_tagged_sents,
                       train_sents,
                       test_sents,
                       test_sents_untagged,
                       simplified_tags)

    # Bigram HMM + Smoothing -> Viterbi Sequence Tagger
    logging.info("Bigram HMM + Add-1 Smoothing --> Viterbi")
    bigram_hmm_viterbi(brown_simplified_tagged_sents,
                       train_sents,
                       test_sents,
                       test_sents_untagged,
                       simplified_tags,
                       smoothing=True)

    # Bigram HMM + Pseudo classes -> Viterbi Sequence Tagger
    logging.info("Bigram HMM + Pseudo Classes --> Viterbi")
    bigram_hmm_viterbi(brown_simplified_tagged_sents,
                       train_sents,
                       test_sents,
                       test_sents_untagged,
                       simplified_tags,
                       pseudo_classes=True)

    # Bigram HMM + Pseudo classes + Smoothing -> Viterbi Sequence Tagger
    logging.info("Bigram HMM + Pseudo Classes + Smoothing --> Viterbi")
    bigram_hmm_viterbi(brown_simplified_tagged_sents,
                       train_sents,
                       test_sents,
                       test_sents_untagged,
                       simplified_tags,
                       smoothing=True,
                       pseudo_classes=True,
                       plot_confusion_matrix=True)


if __name__ == '__main__':
    main()
