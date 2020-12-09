import logging
from collections import Counter
from typing import List, Tuple, Dict

from nltk.corpus import brown


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # http://www.nltk.org/book/ch05.html
    brown_tagged_sents = brown.tagged_sents(categories='news')
    brown_sents = brown.sents(categories='news')
    size = int(len(brown_tagged_sents) * 0.9)
    train_sents = brown_tagged_sents[:size]
    test_sents = brown_tagged_sents[size:]

    word_probabilities = MLETagger(train_sents)
    logging.info(word_probabilities)

    accuracies = getErrorRate(train_sents, test_sents, MLETagger)
    logging.info(accuracies)
