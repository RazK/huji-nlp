from itertools import islice
from data_loader import get_negated_polarity_examples, get_rare_words_examples

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"

BATCH_SIZE = 64  # Instructed by the PDF, page 5
N_EPOCHS = 10  # TODO: Shahaf: Why?
LEARNING_RATE = 0.01  # Instructed by the PDF, page 5
WEIGHT_DECAY = 0.0001  # Instructed by the PDF, page 5

# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict

def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """

    size = len(word_to_vec)
    sum_vector_embeddings = np.zeros(embedding_dim)
    for word in sent.text:
        sum_vector_embeddings += word_to_vec.get(word, np.zeros(embedding_dim))
    return sum_vector_embeddings / len(sent.text)

def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    np.put(one_hot, ind, 1)
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    size = len(word_to_ind)
    sum_one_hots = np.zeros(size)
    for word in sent.text:
        sum_one_hots += get_one_hot(size, word_to_ind[word])
    return sum_one_hots / len(sent.text)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: words_list.index(word) for word in words_list}

def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    word_embedding_lst = []
    for word in sent.text:
        word_embedding_lst.append(word_to_vec.get(word, np.zeros(embedding_dim)))
    if len(word_embedding_lst) >= seq_len:
        return np.array(word_embedding_lst)[:seq_len]
    return np.concatenate((np.array(word_embedding_lst), np.zeros((seq_len-len(word_embedding_lst), embedding_dim))))

class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_size = 64
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.hidden2sent = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        lstm_out, _ = self.lstm(text)
        return self.hidden2sent(lstm_out[:, -1, :])

    def predict(self, text):
        return self.sigmoid(self.forward(text))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = torch.nn.Linear(embedding_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return self.sigmoid(self.forward(x))


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    return (0.5 < preds).type(y.dtype).eq(y.view_as(preds)).sum().item() / \
           len(y)


def train_epoch(model, data_iterator, optimizer, criterion, epoch):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    :param epoch: the number of the current epoch (for logging)
    """
    model.train()
    num_full_batches = int(len(data_iterator.dataset) /
                           data_iterator.batch_size)
    limit = num_full_batches * data_iterator.batch_size

    train_loss = 0
    train_correct = 0
    for batch_idx, (inputs, labels) in islice(enumerate(data_iterator),
                                              num_full_batches):
        optimizer.zero_grad()
        actual = labels.reshape(data_iterator.batch_size, 1).type(
            torch.FloatTensor)
        outputs = model(inputs.type(torch.FloatTensor))
        predictions = model.predict(inputs.type(torch.FloatTensor))
        loss = criterion(outputs, actual)
        loss.backward()
        optimizer.step()

        train_loss += float(loss)
        train_correct += binary_accuracy(predictions, actual) * len(actual)

        if (batch_idx % 100 == 0):
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: '
                '{}/{} ({:.0f}%)'.format(
                    epoch,
                    batch_idx * len(inputs), limit, 100. * batch_idx /
                    num_full_batches, loss.item(),
                    train_correct, (batch_idx + 1) * data_iterator.batch_size,
                    100. * train_correct / ((batch_idx + 1) *
                                            data_iterator.batch_size)))

    return train_loss / limit, train_correct / limit


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over
    all examples)
    """
    model.eval()
    num_full_batches = int(len(data_iterator.dataset) /
                           data_iterator.batch_size)
    limit = num_full_batches * data_iterator.batch_size

    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in islice(enumerate(data_iterator),
                                                  num_full_batches):
            actual = labels.reshape(data_iterator.batch_size, 1).type(
                torch.FloatTensor)
            outputs = model(inputs.type(torch.FloatTensor))
            predictions = model.predict(inputs.type(torch.FloatTensor))
            loss = criterion(outputs, actual)
            test_loss += float(loss)
            test_correct += binary_accuracy(predictions, actual) * len(actual)

    print(
        '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_correct, limit,
            100. * test_correct / limit))
    return test_loss / limit, test_correct / limit


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    model.eval()
    num_full_batches = int(len(data_iter.dataset) /
                           data_iter.batch_size)
    limit = num_full_batches * data_iter.batch_size

    test_correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in islice(enumerate(data_iter),
                                                  num_full_batches):
            actual = labels.reshape(data_iter.batch_size, 1).type(
                torch.FloatTensor)
            predictions = model.predict(inputs.type(torch.FloatTensor))
            test_correct += binary_accuracy(predictions, actual) * len(actual)

    return 100 * test_correct / limit


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    train_loader = data_manager.get_torch_iterator(TRAIN)
    validation_loader = data_manager.get_torch_iterator(VAL)
    test_loader = data_manager.get_torch_iterator(TEST)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    for epoch in range(n_epochs):
        print(epoch)
        train_loss, train_accuracy = train_epoch(model, train_loader,
                                                 optimizer, criterion, epoch)
        validation_loss, validation_accuracy = evaluate(model,
                                                        validation_loader,
                                                        criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

    test_accuracy = get_predictions_for_data(model,test_loader)
    print("Test set: Accuracy: {:.0f}%"
          .format(test_accuracy))

    test_sentences = data_manager.sentences[TEST]
    test_labels = data_manager.get_labels(TEST)
    negated_indexes = get_negated_polarity_examples(test_sentences)
    rare_words_indexes = get_rare_words_examples(test_sentences, data_manager.sentiment_dataset)

    all_word_vectors = []
    for batch in list(data_manager.get_torch_iterator(TEST)):
        for vector in batch[0]:
            all_word_vectors.append(vector)
    negated_inputs = [all_word_vectors[i].float() for i in negated_indexes]
    rare_words_inputs = [all_word_vectors[i].float() for i in rare_words_indexes]

    negated_labels = [torch.tensor(test_labels[i]) for i in negated_indexes]
    rare_words_labels = [torch.tensor(test_labels[i]) for i in rare_words_indexes]


    negated_data = torch.stack(negated_inputs)
    rare_words_data = torch.stack(rare_words_inputs)
    negated_labels = torch.stack(negated_labels)
    rare_words_labels = torch.stack(rare_words_labels)

    negated_dataset = torch.utils.data.TensorDataset(negated_data, negated_labels)
    rare_words_dataset = torch.utils.data.TensorDataset(rare_words_data, rare_words_labels)

    negated_test_loader = torch.utils.data.DataLoader(negated_dataset)
    rare_words_test_loader = torch.utils.data.DataLoader(rare_words_dataset)

    negated_test_accuracy = get_predictions_for_data(model,negated_test_loader)
    rare_words_test_accuracy = get_predictions_for_data(model,rare_words_test_loader)

    print("Negated test set: Accuracy: {:.0f}%"
          .format(negated_test_accuracy))

    print("Rare words test set: Accuracy: {:.0f}%"
          .format(rare_words_test_accuracy))

    plt.title("Training Loss Curve (batch_size={}, lr={})".format(
        train_loader.batch_size, lr))
    plt.plot(range(n_epochs), train_losses, label="Train")
    plt.plot(range(n_epochs), validation_losses, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Accuracy Curve (batch_size={}, lr={})".format(
        train_loader.batch_size, lr))
    plt.plot(range(n_epochs), train_accuracies, label="Train")
    plt.plot(range(n_epochs), validation_accuracies, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager(ONEHOT_AVERAGE, batch_size=BATCH_SIZE)
    model = LogLinear(data_manager.get_input_shape()[0])
    train_model(model, data_manager, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY)

def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(W2V_AVERAGE, batch_size=BATCH_SIZE, embedding_dim=300)
    model = LogLinear(data_manager.get_input_shape()[0])
    train_model(model, data_manager, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY)

def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(W2V_SEQUENCE, batch_size=BATCH_SIZE, embedding_dim=300)
    model = LSTM(data_manager.get_input_shape()[1], 100, 1, 0.5)
    train_model(model, data_manager, 1, 0.001, 0.0001)

if __name__ == '__main__':
    train_log_linear_with_one_hot()
    #train_log_linear_with_w2v()
    #train_lstm_with_w2v()