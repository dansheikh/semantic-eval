# How To Use CRFSuite

crfsuite learn -m [save_model_path] [training_data_path]


crfsuite tag -r -m [trained_model_path] [testing_data_path] > [log_path]

# How to Evaluate Models

conlleval.pl -r -d $'\t' < [log_path]

fmeasure.py [log_path]

# How to Use RNN Models

## Training RNN Model
tf_word2vec_rnn.py -m learn -s [checkpoint_path] [embed_path] [data_path]

## Evaluating RNN Model
tf_word2vec_rnn.py -m eval -l [checkpoint_path] [embed_path] [data_path]
