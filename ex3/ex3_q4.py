import logging
from collections import Counter

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")


class Embedding:
    def __init__(self, input_sentences, remove_words):
        self.input_sentences = input_sentences
        self.remove_words = remove_words

        X_cd, X_nparray = q4_1(input_sentences, remove_words)
        self.X_cd = X_cd
        self.X_nparray = X_nparray

        # Q4-2 Answer:
        u, s, vh = q4_2(X_nparray)
        self.u = u
        self.s = s
        self.vh = vh

        # Q4-3 Answer:
        (X_reduced, u_reduced, s_reduced, vh_reduced) = \
            q4_3(X_nparray, u, s, vh)
        # The reduction to 30% of the singular values cleans noise from the
        # embedding, preserving only  the 'low-frequency', most significant
        # channels.
        self.X_nparray_reduced = X_reduced
        self.u_reduced = u_reduced
        self.s_reduced = s_reduced
        self.vh_reduced = vh_reduced

        all_words = list(X_cd.keys())
        self.svd_embeddings = get_embeddings_dict(all_words, u_reduced)

        eigenValues, eigenVectors = q4_5_2(X_nparray)
        ev_reduced = q4_5_3(X_nparray, eigenVectors)
        self.evd_embeddings = get_embeddings_dict(all_words, ev_reduced)

    def _get_embedding(self, word, embeddings_dict):
        return embeddings_dict[word]

    def get_embedding(self, word, embedding='svd'):
        embeddings_dict = self.svd_embeddings
        if embedding == 'evd':
            embeddings_dict = self.evd_embeddings
        return self._get_embedding(word, embeddings_dict)

    def get_similarity(self, word1, word2, embedding='svd'):
        return np.dot(self.get_embedding(word1, embedding),
                      self.get_embedding(word2, embedding))


def clean_sentence(sentence, remove_words):
    for char in remove_words:
        sentence = sentence.replace(char, ' ')
    sentence = sentence.replace('.', '')
    return sentence


def counters_dict_2_nparray(counters_dict):
    n = len(counters_dict)
    array = np.zeros((n, n))
    for i, word_1 in enumerate(counters_dict):
        for j, word_2 in enumerate(counters_dict):
            array[i, j] = counters_dict[word_1][word_2]

    return array


def q4_1(input_sentences, remove_words):
    stripped_input = [clean_sentence(sentence, remove_words).lower() for
                      sentence in input_sentences]

    # Matrix where X[i][j] = X[j][i] = #(x_i, x_j) appear consecutively in the
    # sentence (window size is 1 so we look at immediate neighbours)
    X = {word: Counter()
         for sentence in stripped_input
         for word in sentence.split(' ')}
    logging.debug(X)

    for sentence in stripped_input:
        words = sentence.split(' ')
        for i in range(len(words) - 1):
            X[words[i]][words[i + 1]] += 1
            X[words[i + 1]][words[i]] += 1
            X[words[i]][words[i]] = 0
            X[words[i + 1]][words[i + 1]] = 0
    logging.debug(X)

    # pretty print
    header = ' ' * 10
    for word in X:
        header += "{:<10}".format(word)
    logging.info(header)

    for i in X:
        line = "{:<10}".format(i)
        for j in X:
            line += "{:<10}".format(X[i][j])
        logging.info(line)

    return X, counters_dict_2_nparray(X)


def q4_2(X):
    u, s, vh = np.linalg.svd(X)
    ru = np.around(u, 1)
    rs = np.around(s, 1)
    rvh = np.around(vh, 1)
    logging.info("Matrix X:\n{}\n".format(X))
    logging.info("Matrix U:\n{}\n".format(ru))
    logging.info("Matrix S:\n{}\n".format(np.diag(rs)))
    logging.info("Matrix Vh:\n{}\n".format(rvh))
    return (u, s, vh)


def q4_3(X, u, s, vh):
    thirty_percent = int(0.3 * len(X[0]))
    u_reduced = u[:, :thirty_percent]
    logging.info("Matrix U Reduced ({}):\n{}\n".format(
        np.shape(u_reduced),
        np.around(u_reduced, 1)))

    s_reduced = s[:thirty_percent]
    logging.info("Matrix S Reduced ({}):\n{}\n".format(
        np.shape(s_reduced),
        np.around(np.diag(s_reduced), 1)))

    vh_reduced = vh[:thirty_percent, :]
    logging.info("Matrix Vh ({}):\n{}\n".format(
        np.shape(vh_reduced),
        np.around(vh_reduced, 1)))

    return (np.dot(np.dot(u_reduced, np.diag(s_reduced)), vh_reduced)), \
           u_reduced, s_reduced, vh_reduced


def get_embeddings_dict(words, embeddings_matrix):
    """
    Get word embeddings as a dictionary.
    :param words: list of words, same order as embeddings.
    :param embeddings: np.array, first axis is ordered same as words.
    :return: dict(word : embedding vector)
    """
    raw_embedding = lambda word: embeddings_matrix[words.index(word)]
    embedding = lambda word: raw_embedding(word) / np.linalg.norm(
        raw_embedding(word))
    return {word: embedding(word) for word in words}


def q4_4(X_cd, U_reduced):
    embeddings = get_embeddings_dict(list(X_cd.keys()), U_reduced)

    logging.debug(embeddings)
    logging.info('SVD Cosine Similarities:')
    john_he = np.dot(embeddings['john'], embeddings['he'])
    logging.info('John-He: {}'.format(john_he))

    john_subfield = np.dot(embeddings['john'], embeddings['subfield'])
    logging.info('John-Subfield: {}'.format(john_subfield))

    deep_machine = np.dot(embeddings['deep'], embeddings['machine'])
    logging.info('Deep-Machine: {}'.format(deep_machine))

    return embeddings


def q4_5_2(X):
    eigenValues, eigenVectors = np.linalg.eigh(X)
    # idx = eigenValues.argsort()[::-1]
    # eigenValues = eigenValues[idx]
    # eigenVectors = eigenVectors[:, idx]
    eigenValues, eigenVectors = np.flip(eigenValues, axis=0), np.flip(
        eigenVectors,
                                                              axis=1)
    rw = np.around(eigenValues, 1)
    rv = np.around(eigenVectors, 1)
    logging.debug("Eigenvectors:")
    for i in range(len(eigenValues)):
        logging.debug("{} :\n{}\n".format(rv[i], rw[i]))
    return eigenValues, eigenVectors


def q4_5_3(X, sorted_eigenvectors):
    thirty_percent = int(0.3 * len(X[0]))
    ev_reduced = sorted_eigenvectors[:, :thirty_percent]
    logging.info("Eigenvectors Reduced ({}):\n{}\n".format(
        np.shape(ev_reduced),
        np.around(ev_reduced, 1)))

    return (ev_reduced)


def q4_5_4(X_cd, ev_reduced):
    embeddings = get_embeddings_dict(list(X_cd.keys()), ev_reduced)

    logging.debug(embeddings)
    logging.info('EVD Cosine Similarities:')
    john_he = np.dot(embeddings['john'], embeddings['he'])
    logging.info('John-He: {}'.format(john_he))

    john_subfield = np.dot(embeddings['john'], embeddings['subfield'])
    logging.info('John-Subfield: {}'.format(john_subfield))

    deep_machine = np.dot(embeddings['deep'], embeddings['machine'])
    logging.info('Deep-Machine: {}'.format(deep_machine))

    return embeddings


def q4_5(X, X_cd):
    eigenValues, eigenVectors = q4_5_2(X)
    ev_reduced = q4_5_3(X, eigenVectors)
    q4_5_4(X_cd, ev_reduced)


def q4_6(svd_embeddings):
    logging.info("SVD Cosine Similarities:\n"
                 "Wrote-Post: {}".format(np.dot(svd_embeddings['wrote'],
                                                svd_embeddings['post'])))


def q4_7(svd_embeddings):
    logging.info("SVD Cosine Similarities:\n"
                 "Likes-Likes: {}".format(np.dot(svd_embeddings['likes'],
                                                 svd_embeddings['likes'])))


if __name__ == '__main__':
    input_sentences = ["John likes NLP.",
                       "He likes Marry.",
                       "John likes machine learning.",
                       "Deep learning is a subfield of machine learning.",
                       "John wrote a post about NLP and got likes."]
    remove_words = {' is ', ' a ', ' of '}

    # Q4-1 Answer:
    logging.info('\nQuestion 4 - 1\n--------------')
    X_cd, X_nparray = q4_1(input_sentences, remove_words)

    # Q4-2 Answer:
    logging.info('\nQuestion 4 - 2\n--------------')
    u, s, vh = q4_2(X_nparray)

    # Q4-3 Answer:
    logging.info('\nQuestion 4 - 3\n--------------')
    (X_reduced, u_reduced, s_reduced, vh_reduced) = q4_3(X_nparray, u, s, vh)
    logging.info(
        "The reduction to 30% of the singular values cleans noise from the\n"
        "embedding, preserving only the information about the most "
        "significant\n"
        "co-occurrences. This eliminates correlations that don't reflect\n"
        "true relationships (noise), improving the accuracy of the\n"
        "predictions.\n"
    )

    # Q4-4 Answer:
    logging.info('\nQuestion 4 - 4\n--------------')
    svd_embeddings = q4_4(X_cd, u_reduced)
    logging.info("\n"
                 "The embeddings indeed capture a semantic relationship: the "
                 "words \n"
                 "'John' and 'He' are scored 0.996... which corresponds to "
                 "their similar\n"
                 "semantic part in the sentence. Additionally, the words "
                 "'John' and \n"
                 "'Subfield' which are completely unrelated in the sentence "
                 "are scored \n"
                 "poorly in their similarity, as expected."
                 )

    # Q4-5 Answer:
    logging.info('\nQuestion 4 - 5\n--------------')
    evd_embeddings = q4_5(X_nparray, X_cd)
    logging.info("\n"
                 "The results are slightly worse for John-He (bad), "
                 "far worse for\n"
                 "John-Subfield (good) and slightly better for Deep-Machine "
                 "(good).\n"
                 "EVD yields negative values, and the vectors are not "
                 "necessarily\n"
                 "orthogonal - thus yielding results inferior to SVD."
                 )
    # TODO: Shahaf - Explain why

    # Q4-6 Answer:
    logging.info('\nQuestion 4 - 6\n--------------')
    q4_6(svd_embeddings)
    logging.info("\n"
                 "To fix the negative cosine similarity, remove 'a' from the\n"
                 "remove_words list and then 'wrote' and 'post' will appear "
                 "adjacently\n"
                 "in the text once, increasing their similarity "
                 "dramatically:\n"
                 "Wrote-Post: 0.878566731974979"
                 )

    # Q4-7 Answer:
    logging.info('\nQuestion 4 - 7\n--------------')
    q4_7(svd_embeddings)
    logging.info("\n"
                 "The similarity is 1, but that may not always be the case! "
                 "the word\n"
                 "'likes' can sometimes refer to someone's affection towards "
                 "something,\n"
                 "but other times to a noun invented by facebook to control "
                 "our minds\n"
                 "and manipulate us to click advertisements.\n"
                 "To differentiate, we could use POS together with the "
                 "embeddings to\n"
                 "differentiate between the good likes and the bad ones."
                 )

    #Q4-8 Answer:
    logging.info('\nQuestion 4 - 7\n--------------')


    # Q4-8 Answer:
    # TODO: Shahaf - Explain skip-gram
