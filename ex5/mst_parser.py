import logging
from collections import Counter, namedtuple

from nltk import DependencyGraph
from nltk.corpus import dependency_treebank

from ex5.Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx

logging.basicConfig(level=logging.INFO, format="%(message)s")

Arc = namedtuple('Arc', 'head tail')
WeightedArc = namedtuple('WeightedArc', 'head tail weight')


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


def get_feature_function(sentence: DependencyGraph) -> Counter:
    """
    Calculate and return the feature functions over the given sentence.
    The feature function is f:V^2xS->{0,1}^d and encodes the following
    features:

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

    :param sentence: DependencyGraph - nodes and their respective tags
    :return: Counter (word_bigrams | pos_bigrams)
    """
    logging.debug('Building the feature function for sentence of length {'
                  '}...'.format(
        len(sentence.nodes)))
    word_bigrams = Counter()
    pos_bigrams = Counter()
    nodes = sentence.nodes

    # Handle ROOT node
    word_bigrams[('ROOT', nodes[1]['word'])] = 1
    word_bigrams[('ROOT', nodes[1]['tag'])] = 1

    # Handle all following nodes
    for i in range(1, len(nodes) - 1):
        word_bigrams[(nodes[i]['word'], nodes[i + 1]['word'])] = 1
        pos_bigrams[(nodes[i]['tag'], nodes[i + 1]['tag'])] = 1

    logging.debug('word_bigrams: {}'.format(word_bigrams))
    logging.debug('pos_bigrams: {}'.format(pos_bigrams))
    return (word_bigrams | pos_bigrams)


def mst_to_feature_function(mst):
    """
    Convert an MST to a feature function.
    :param mst: dict {v : WeightedArc(head=u, tail=v, weight=score}
    :return: Counter {(u, v) : score}
    """
    features = Counter()
    for arc in mst.values():
        features[(arc.head, arc.tail)] = arc.weight
    return features


def score(weights, feature_function):
    """
    Calculate a score graph for the given tree with the given arc weights
    and feature functions.
    :param weights: dict {(u, v) : weight}
    :param feature_function: Counter {(u, v) : f(u,v)}
    :return: List [(head, tail, score)] where score is calculated by
    score = f(head, tail) * weights[(head, tail)]
    """
    arcs = []
    for arc in feature_function.keys():
        arcs.append(WeightedArc(head=arc[0],
                                tail=arc[1],
                                weight=feature_function[arc] * weights[arc]))
    return arcs


def get_mst(weights, sentence_feature_function):
    """
    Given a sentence and a weights vector, predict an MST for that sentence.
    :param sentence: DependencyGraph
    :param weights: Counter {(u, v) : weight}
    :return: MST {tail : WeightedArc(head, tail, weight)}
    """
    return min_spanning_arborescence_nx(
        score(weights, sentence_feature_function))


def mst_index_arcs(mst, sentence):
    """
    Convert an MST holding nodes, to an MST holding their indices in a
    given sentence.
    :param sentence: DependencyGraph
    :param mst: {tail : WeightedArc(head, tail, weight)}
    :return: arcs = set {Arc(head_idx, tail_idx)}
    """
    node2index = {frozenset(node): index for index, node in
               sentence.nodes.items()}
    index_arcs = set()
    for arc in mst.values():
        head = frozenset(arc['head'])
        tail = frozenset(arc['tail'])
        head_idx = node2index[head]
        tail_idx = node2index[tail]
        index_arcs.add(Arc(head_idx, tail_idx))
    return index_arcs

def averaged_perceptron(sentences, N_iterations=2, learning_rate=1.0):
    logging.debug('Running Averaged Perceptron...')
    weights = Counter()
    N = len(sentences) * N_iterations
    for r in range(N_iterations):
        logging.debug("Iteration {}...".format(r))
        for i, sentence in enumerate(sentences):
            logging.info("Training... [{:4d} / {:4d}]".format(
                len(sentences) * r + i, N))
            logging.debug("Sentence: {}".format(sentence))
            sentence_feature_function = get_feature_function(sentence)
            mst = get_mst(weights, sentence_feature_function)
            logging.debug("MST: {}".format(mst))
            index_arcs = mst_index_arcs(mst, sentence)
            mst_feature_function = mst_to_feature_function(mst)
            logging.debug(
                "MST Feature Function: {}".format(mst_feature_function))
            weights_update = (sentence_feature_function - mst_feature_function)
            for k in weights_update.keys():
                weights_update[k] *= learning_rate
            weights += weights_update
    # Normalize and return weights
    for k in weights.keys():
        weights[k] /= (len(sentences) * N_iterations)
    return weights


def ground_truth(sentence):
    """
    Extract the ground truth MST from the given sentence
    :param sentence: DependencyGraph
    :return: Sentence Arcs = set {Arc(head=u_idx, tail=v_idx})
    """
    return set(Arc(node["head"], node["address"]) for node in
               sentence.nodes.values() if node["head"])


def evaluate(dataset, weights):
    for sentence in dataset:
        feature_function = get_feature_function(sentence)
        mst = get_mst(weights, feature_function)
        arcs = mst_index_arcs(mst)
        ground_truth = ground_truth(sentence)
        # compare arcs to ground_truth


if __name__ == "__main__":
    # Load Dataset
    logging.info('Loading dataset...')
    dl = DataLoader()
    logging.info('Train set: {} sentences'.format(len(dl.get_train_set())))
    logging.info('Validation set: {} sentences'.format(
        len(dl.get_validation_set())))
    logging.info('Test set: {} sentences'.format(len(dl.get_test_set())))
    logging.info('Total: {} sentences'.format(len(dl.sentences)))

    # Test get_feature_function
    sentence = dl.get_train_set()[0]
    # feature_function = get_feature_function(sentence)
    # logging.info('Sentence Feature Function: {}'.format(feature_function))
    gt = ground_truth(sentence)

    # Run averaged_perceptron on the train dataset
    trained_weights = averaged_perceptron(dl.get_train_set())

    # Evaluate using trained weights
    evaluate(dl.get_train_set(), trained_weights)
