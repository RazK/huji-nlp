import os
import random


POSITIVE_SENTIMENT = 1.
NEGATIVE_SENTIMENT = 0.
NEUTRAL_SENTIMENT = -1.


SENTIMENT_NAMES = {
    POSITIVE_SENTIMENT: "Positive",
    NEUTRAL_SENTIMENT: "Neutral",
    NEGATIVE_SENTIMENT: "Negative"
}
SENTS_PATH = "SOStr.txt"
TREES_PATH = "STree.txt"
DICT_PATH = "dictionary.txt"
LABELS_path = "sentiment_labels.txt"


def get_sentiment_class_from_val(sentiment_val: float):
    if sentiment_val <= 0.4:
        return NEGATIVE_SENTIMENT
    elif sentiment_val >= 0.6:
        return POSITIVE_SENTIMENT
    else:
        return NEUTRAL_SENTIMENT

class SentimentTreeNode(object):
    def __init__(self, text: list, sentiment_val: float, min_token_idx: int, children=[], parent=None):
        self.text = text
        self.sentiment_val = sentiment_val
        self.min_token_idx = min_token_idx
        self.sentiment_class = get_sentiment_class_from_val(sentiment_val)
        self.children = children
        self.parent = parent


class Sentence(object):
    """
    Represents a sentence in sentiment tree bank.
    You can access the sentence text by sent.text
    This will give you a list of tokens (strings) in the order that they appear in the sentence.
    sent.sentiment_class is the coding of the annotated sentiment polarity of the sentence.
    sent.sentiment_val is the exact annotated sentiment value in the range [0,1]
    """
    def __init__(self, sentence_root: SentimentTreeNode):
        self.root = sentence_root
        self.text = sentence_root.text
        self.sentiment_class = sentence_root.sentiment_class
        self.sentiment_val = sentence_root.sentiment_val

    def _get_leaves_recursively(self, cur_root):
        if len(cur_root.children) == 0:
            return [cur_root]
        else:
            cur_leaves = []
            for child in cur_root.children:
                cur_leaves += self._get_leaves_recursively(child)
            return cur_leaves

    def get_leaves(self):
        return self._get_leaves_recursively(self.root)

    def __repr__(self):
        return " ".join(self.text) + " | " + SENTIMENT_NAMES[self.sentiment_class] + " | " + str(self.sentiment_val)


class SentimentTreeBank(object):
    """
    The main object that represents the stanfordSentimentTreeBank dataset. Can be used to access the
    examples and some other utilities.
    """
    def __init__(self, path="stanfordSentimentTreebank", split_ratios=(0.8,0.1,0.1), split_words=True):
        """

        :param path: relative or absolute path to the datset directory
        :param split_ratios: split ratios for train, validation and test. please do not change!
        :param split_words: whether to split tokens with "-" and "/" symbols. please do not change!
        """
        self._base_path = path
        self.split_words = split_words
        sentences = self._read_sentences()
        self.sentences = self._build_dataset(sentences)
        if self.split_words:
            for sent in self.sentences:
                leaves = sent.get_leaves()
                for node in leaves:
                    node_text = node.text
                    splitted = node_text[0].split("-")
                    splitted_final = []
                    for s in splitted:
                        splitted_final.extend(s.split("\\/"))
                    if len(splitted_final) > 1 and all([len(s) > 0 for s in splitted_final]):
                        leaves = [SentimentTreeNode([s], node.sentiment_val,
                                                    min_token_idx=node.min_token_idx, parent=node) for
                                  s in splitted_final]
                        node.text = splitted_final
                        node.children = leaves
                        cur_parent = node.parent
                        while cur_parent != None:
                            cur_parent.text = []
                            for child in cur_parent.children:
                                cur_parent.text.extend(child.text)
                            cur_parent = cur_parent.parent
                sent.text = sent.root.text


        assert len(split_ratios) == 3
        assert sum(split_ratios) == 1
        self.split_ratios = split_ratios



    def _read_sentences(self):
        sentences = []
        with open(os.path.join(self._base_path, SENTS_PATH), "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for i, line in enumerate(lines):
                if len(line.strip()) == 0:
                    continue
                line_content = line.strip()
                tokens = line_content.split("|")
                tokens = [t.lower().replace("-lrb-","(").replace("-rrb-", ")") for t in tokens]
                sentences.append(tokens)
        return sentences

    def _build_dataset(self, sentences):
        phrases_dictionary = {}

        # extract phrases
        with open(os.path.join(self._base_path, DICT_PATH), "r", encoding="utf-8") as f:
            lines = f.read().split("\n")[:-1]
            for line in lines:
                phrase, phrase_id = line.strip().split("|")
                phrases_dictionary[phrase.lower().replace("-lrb-","(").replace("-rrb-", ")")] = int(phrase_id)

        # extract labels
        with open(os.path.join(self._base_path, LABELS_path), "r",  encoding="utf-8") as f:
            lines = [l.strip().split("|") for l in f.read().split("\n")[1:-1]]
            labels_dict = {int(l[0]): float(l[1]) for l in lines}


        def get_val_from_phrase(phrase_tokens_list):
            try:
                return labels_dict[phrases_dictionary[" ".join(phrase_tokens_list)]]
            except:
                print("couldn't find key!")

        # load the sentences tree structures
        tree_pointers = []
        with open(os.path.join(self._base_path, TREES_PATH), "r") as f:
            for line in f.readlines():
                sent_pointers = [int(p) for p in line.strip().split("|")]
                tree_pointers.append(sent_pointers)
        assert len(tree_pointers) == len(sentences)

        # create Sentence instances with tree of SentimentTreeNodes
        labeled_sentences = []
        for sent, sent_pointers in zip(sentences, tree_pointers):
            try:

                children_dict = {i: [] for i in range(len(sent_pointers))}
                for i, p in enumerate(sent_pointers):
                    if i < len(sent):
                        node_text = [sent[i]]
                        node = SentimentTreeNode(text=node_text, sentiment_val=get_val_from_phrase(node_text),
                                                 min_token_idx=i)
                    else:
                        children = children_dict[i]
                        children = sorted(children, key= lambda n: n.min_token_idx)
                        node_text = []
                        for child in children:
                            node_text.extend(child.text)
                        node = SentimentTreeNode(text=node_text, sentiment_val=get_val_from_phrase(node_text),
                                                 children=children, min_token_idx=children[0].min_token_idx)
                        for child in children:
                            child.parent = node
                    if p > 0:
                        children_dict[p - 1].append(node)
                    last_node = node
                new_sentence = Sentence(last_node)
                if new_sentence.sentiment_class == NEUTRAL_SENTIMENT:
                    continue
                labeled_sentences.append(new_sentence)
            except Exception as e:
                raise e
                print("couldn't parse sentence!")
                print(sent)
        random.Random(1).shuffle(labeled_sentences) # shuffle but with the same shuffle each time
        return labeled_sentences

    def get_train_set(self):
        """
        :return: list of Sentence instances for the train part of the dataset
        """
        if not hasattr(self, "_train_set"):
            self._train_set = self.sentences[:int(self.split_ratios[0] * len(self.sentences))]
        return self._train_set

    def _extract_all_phrases(self, root):
        phrases = [Sentence(root)] if root.sentiment_class != NEUTRAL_SENTIMENT else []
        if len(root.text) == 1:
            return []
        for child in root.children:
            phrases += self._extract_all_phrases(child)
        return phrases

    def get_train_set_phrases(self):
        """
        :return: list of Sentence instances for the train part of the dataset including all sub-phrases
        """
        if not hasattr(self, "_train_set_phrases"):
            train_set = self.get_train_set()
            train_set_phrases = []
            for sent in train_set:
                train_set_phrases += self._extract_all_phrases(sent.root)
            self._train_set_phrases = train_set_phrases
        return self._train_set_phrases

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
        :return: list of Sentence instances for the validation part of the dataset
        """
        if not hasattr(self, "_validation_set"):
            self._validation_set = self.sentences[int(sum(self.split_ratios[:2]) * len(self.sentences)):]
        return self._validation_set

    def get_train_word_counts(self):
        """
        :return: dictionary of all words in the train set with their frequency in the train set.
        """
        if not hasattr(self, "_train_word_counts"):
            word_counts = {}
            for sent in self.get_train_set():
                for word_node in sent.get_leaves():
                    assert len(word_node.text) == 1
                    word_text = word_node.text[0]
                    if word_text in word_counts:
                        word_counts[word_text] += 1
                    else:
                        word_counts[word_text] = 1
            self._train_word_counts = word_counts

        return self._train_word_counts

    def get_word_counts(self):
        """
        :return: dictionary of all words in the dataset with their frequency in the whole dataset.
        """
        if not hasattr(self, "_word_counts"):
            word_counts = {}
            for sent in self.sentences:
                for word_node in sent.get_leaves():
                    assert len(word_node.text) == 1
                    word_text = word_node.text[0]
                    if word_text in word_counts:
                        word_counts[word_text] += 1
                    else:
                        word_counts[word_text] = 1
            self._word_counts = word_counts

        return self._word_counts



def get_negated_polarity_examples(sentences_list, num_examples=None, choose_random=False):
    """
    Returns the indices of the sentences in sentences_list which have subphrase in the second level with
    sentiment polarity different than the whole sentence polarity.
    :param sentences_list: list of Sentence objects
    :param num_examples: number of examples to return, if None all of them are returned
    :param choose_random: relevant only if num_examples is lower than the number of exisitng negated
    polarity examples in sentences_list
    """

    if num_examples is None:
        num_examples = len(sentences_list) # take all possible sentences

    def is_polarized(sent: Sentence):
        if sent.sentiment_class == NEUTRAL_SENTIMENT:
            return False
        else:
            root_polarity = sent.sentiment_class
            for child in sent.root.children:
                if child.sentiment_class == 1 - root_polarity:
                    return True
            return False
    indexed_senteces = list(enumerate(sentences_list))
    negated_sentences = list(filter(lambda s: is_polarized(s[1]), indexed_senteces))
    negated_sentences_indices = [i for i,s in negated_sentences]
    if len(negated_sentences) <= num_examples:
        return negated_sentences_indices
    else:
        if choose_random:
            random.shuffle(negated_sentences_indices)
        return negated_sentences_indices[:num_examples]


def get_sentiment_words(sent: Sentence):
    sent_polarity = sent.sentiment_class
    return [node for node in sent.get_leaves() if node.sentiment_class == sent_polarity]


def get_rare_words_examples(sentences_list, dataset: SentimentTreeBank,
                            num_sentences=50):
    """
    Computes for each sentence in sentences the maximal train frequency of sentiment word, where sentiment
    word is a word which is labeled with either positive or negative sentiment value, and returns the
    indices of the <num_sentences> sentences with lowest value.
    :param sentences_list: list of Sentence objects
    :param dataset: the SentimentTreebank datset object
    :param num_sentences: number of sentences to return
    :return: list of ints representing the indices of the chosen sentences out of the input sentences_list
    """
    word_counts = dataset.get_train_word_counts()

    def get_count(word_node: SentimentTreeNode):
        word_text = word_node.text[0]
        if word_text in word_counts:
            return word_counts[word_text]
        else:
            return 0
    indexed_sentences = list(enumerate(sentences_list))
    indexed_sentences = list(filter(lambda s: len(get_sentiment_words(s[1])) > 0, indexed_sentences))
    indexed_sentences = sorted(indexed_sentences, key= lambda s: max([get_count(node) for node in
                                                                      get_sentiment_words(s[1])]))
    indices = [i for i,s in indexed_sentences]
    return indices[:num_sentences]




if __name__ == '__main__':
    # examples for reading the sentiment dataset
    dataset = SentimentTreeBank()
    # get train set
    print(dataset.get_train_set()[:2])
    print(dataset.get_train_set()[0].sentiment_val)
    # get word counts dictionary
    print(list(dataset.get_word_counts().keys())[:10])
