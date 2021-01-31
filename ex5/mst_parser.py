import logging
from collections import Counter

from nltk.corpus import dependency_treebank

logging.basicConfig(level=logging.INFO, format="%(message)s")


class DataLoader:
    def __init__(self, split_ratios=(0.9, 0.1, 0.0)):
        """
        Initialize the train, validation and test set
        :param split_ratios: (train, test, validation)
        """
        self.split_ratios = split_ratios
        self.sentences = dependency_treebank.parsed_sents()

    def get_train_set(self):
        """
        :return: list of Sentence instances for the train part of the dataset
        """
        if not hasattr(self, "_train_set"):
            self._train_set = self.sentences[
                              :int(self.split_ratios[0] * len(self.sentences))]
        return self._train_set

    def get_test_set(self):
        """
        :return: list of Sentence instances for the test part of the dataset
        """
        if not hasattr(self, "_test_set"):
            begin_index = int(self.split_ratios[0] * len(self.sentences))
            end_index = int(sum(self.split_ratios[:2]) * len(self.sentences))
            self._test_set = self.sentences[begin_index:end_index]
        return self._test_set

    def get_validation_set(self):
        """
        :return: list of Sentence instances for the validation part of the
        dataset
        """
        if not hasattr(self, "_validation_set"):
            self._validation_set = self.sentences[int(
                sum(self.split_ratios[:2]) * len(self.sentences)):]
        return self._validation_set

    def get_bigram_graphs(self, sentence):
        """
        Word Bigrams
        ------------
        For a potential edge between the nodes u, v ∈ V , the feature
        function will have a feature for every pair of word forms (types) w,
        w0, which has a value of 1 if the node u is the word w and the node
        v is the word w0

        POS Bigrams
        -----------
        For a potential edge between the nodes u, v ∈ V , the feature
        function will have a feature for every pair of POS tags t, t0 ,
        which has a value of 1 if the node u has the POS tag t and the node
        v has the POS tag t0

        :param sentence:
        :return:
        """
        logging.info(
            'Building graphs for sentence of length {}...'.format(
                len(sentence.nodes)))
        word_bigrams = Counter()
        pos_bigrams = Counter()
        nodes = sentence.nodes
        for i in range(len(nodes) - 1):
            word_bigrams[(nodes[i]['word'], nodes[i + 1]['word'])] = 1
            pos_bigrams[(nodes[i]['tag'], nodes[i + 1]['tag'])] = 1
        logging.info('word_bigrams: {}'.format(word_bigrams))
        logging.info('pos_bigrams: {}'.format(pos_bigrams))
        return word_bigrams, pos_bigrams

if __name__ == "__main__":
    logging.info('Loading dataset...')
    dl = DataLoader()
    logging.info('Train set: {} sentences'.format(len(dl.get_train_set())))
    logging.info('Validation set: {} sentences'.format(
        len(dl.get_validation_set())))
    logging.info('Test set: {} sentences'.format(len(dl.get_test_set())))
    logging.info('Total: {} sentences'.format(len(dl.sentences)))

    sentence = dl.get_train_set()[0]
    word_bigrams, pos_bigrams = dl.get_bigram_graphs(sentence)