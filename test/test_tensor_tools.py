import numpy as np
import unittest
import utilities.tensor_tools as tt


class TestTensorTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._label_dict = {'O': 0, 'Material': 1, 'Process': 2, 'Task': 3}

    @classmethod
    def tearDownClass(cls):
        cls._label_dict = None

    def test_preprocess(self):
        (sequences, labels) = tt.preprocess('embeds/skipgram_words_vec', 'data/semeval2017_debug_keyphrases_singlabel.conll.txt')
        seq_shape = np.shape(sequences)
        lbl_shape = np.shape(labels)
        self.assertEqual(seq_shape, lbl_shape)

    def test_one_hot(self):
        (sequences, labels) = tt.preprocess('embeds/skipgram_words_vec', 'data/semeval2017_debug_keyphrases_singlabel.conll.txt')
        one_hot = tt.one_hot(labels, self._label_dict)
        self.assertEqual((35, 4), np.shape(one_hot[0]))
        self.assertEqual((16, 4), np.shape(one_hot[1]))
        self.assertTrue(np.array_equal(one_hot[0][0], [0, 0, 0, 1]))
        self.assertTrue(np.array_equal(one_hot[1][0], [1, 0, 0, 0]))

    def test_package_batch(self):
        PAD = np.zeros(100)
        (sequences, labels) = tt.preprocess('embeds/skipgram_words_vec', 'data/semeval2017_debug_keyphrases_singlabel.conll.txt')
        one_hot = tt.one_hot(labels, self._label_dict)
        (seq_len, seq_lbl_zip) = tt.package_batch([0, 1], sequences, one_hot, self._label_dict)
        (batch_x, batch_y) = zip(*seq_lbl_zip)
        self.assertEqual((2, 35, 100), np.shape(batch_x))
        self.assertEqual((2, 35, 4), np.shape(batch_y))
        self.assertTrue(np.array_equal(batch_x[1][16], PAD))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestTensorTools('test_preprocess'))
    suite.addTest(TestTensorTools('test_one_hot'))
    suite.addTest(TestTensorTools('test_package_batch'))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
