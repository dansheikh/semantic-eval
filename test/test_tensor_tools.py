import unittest
import utilities.tensor_tools as tt


class TestTensorTools(unittest.TestCase):

    def test_preprocess(self):
        (sequences, labels) = tt.preprocess('embeds/skipgram_words_vec', 'data/semeval2017_train_keyphrases_singlabel.conll.txt')
        seq_len = len(sequences)
        msg = "Expected sequence length is {expected}, but actual sequence length is {actual}".format(expected=2357, actual=seq_len)
        self.assertEqual(seq_len, 2270, msg)
