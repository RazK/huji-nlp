import unittest

from exercise_blanks import *

class TestExerciseBlanks(unittest.TestCase):
    def test_get_one_hot(self):
        expected = np.array([0,0,0,1,0])
        result = get_one_hot(5,3)
        np.testing.assert_array_equal(result, expected)

    def test_get_word_to_ind(self):
        data = ['a', 'b', 'c', 'a']
        expected = {'a' : 0, 'b' : 1, 'c' : 2}
        result = get_word_to_ind(data)
        self.assertEqual(result, expected)

    def test_average_one_hots(self):
        vocab = ['a', 'b', 'c', 'a']
        word_to_ind = get_word_to_ind(vocab)
        sentence = ['a', 'a', 'c', 'a', 'b']
        expected = [0.6, 0.2, 0.2]
        result = average_one_hots(sentence, word_to_ind)
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()