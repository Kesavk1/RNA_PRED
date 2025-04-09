from tensorflow.keras import Model
#from keras_nlp.layers import TransformerDecoder, TransformerEncoder, PositionEmbedding
from tensorflow.keras import Sequential
from itertools import repeat
from tensorflow.keras.layers import Dropout
from transformers import TFAutoModel

class RNA(Model):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = TFAutoModel.from_pretrained('AmelieSchreiber/esm2_t6_8M_UR50D_rna_binding_site_predictor')#Sequential([*repeat(TransformerEncoder(hidden_dim*4, 6, activation='gelu', dropout=0.1, normalize_first=True), 12)])
        self.dp = Dropout(0.2)
        self.fc = tf.keras.layers.Dense(1)
    def call(self, x):
        x = self.encoder(x).last_hidden_state
        return tf.squeeze(self.fc(x), -1)

#
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf

tokenizer = TextVectorization(
    output_mode='int',
    ngrams=1,
    output_sequence_length=433,
    split='character',
    vocabulary=['a', 'c', 'g', 'u']
)

#
import tensorflow as tf

def loss_fn(labels, targets):
    labels_mask = tf.math.is_nan(labels)
    labels = tf.where(labels_mask, tf.zeros_like(labels), labels)
    mask_count = tf.math.reduce_sum(tf.where(labels_mask, tf.zeros_like(labels), tf.ones_like(labels)))
    loss = tf.math.abs(labels - targets)
    loss = tf.where(labels_mask, tf.zeros_like(loss), loss)
    loss = tf.math.reduce_sum(loss)/(mask_count if mask_count != 0.0 else 1.0)
    return loss

#
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Dense, ReLU, Flatten, Softmax, SimpleRNN, LSTM, MultiHeadAttention
import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, SimpleRNNCell, RNN, LSTM, Bidirectional, LSTMCell
from sklearn.model_selection import KFold
import numpy as np

with strategy.scope():
    model_a3 = RNA(192)
    model_a3.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                loss=loss_fn,
            )

    train_dataset = tf.data.Dataset.from_tensor_slices((x_a3.values, y_a3.values)).batch(128).map(lambda x, y: (tokenizer(x), tf.clip_by_value(y, 0, 1)))
    train_dataset = train_dataset.shuffle(train_dataset.cardinality())
    train_split = train_dataset.take(int(len(train_dataset)*0.8))
    val_split = train_dataset.skip(int(len(train_dataset)*0.8)).take(int(len(train_dataset)*0.2))
    model_a3.fit(train_split, validation_data=val_split, epochs=25, batch_size=128)

#
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Dense, ReLU, Flatten, Softmax, SimpleRNN, LSTM, MultiHeadAttention
import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, SimpleRNNCell, RNN, LSTM, Bidirectional, LSTMCell
from sklearn.model_selection import KFold
import numpy as np

with strategy.scope():
    model_dms = RNA(192)
    model_dms.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                loss=loss_fn,
            )
    train_dataset = tf.data.Dataset.from_tensor_slices((x_dms.values, y_dms.values)).batch(128).map(lambda x, y: (tokenizer(x), tf.clip_by_value(y, 0, 1)))
    train_dataset = train_dataset.shuffle(train_dataset.cardinality())

    train_split = train_dataset.take(int(len(train_dataset)*0.8))
    val_split = train_dataset.skip(int(len(train_dataset)*0.8)).take(int(len(train_dataset)*0.2))
    model_dms.fit(train_split, validation_data=val_split, epochs=25, batch_size=128)

#
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf

tokenizer = TextVectorization(
    output_mode='int',
    ngrams=1,
    output_sequence_length=457,
    split='character',
    vocabulary=['a', 'c', 'g', 'u']
)

test = read_csv('/kaggle/input/stanford-ribonanza-rna-folding/test_sequences.csv')

with strategy.scope():
    predictions_a3 = model_a3.predict(tf.data.Dataset.from_tensor_slices((test.sequence)).batch(128).map(lambda x: tokenizer(x)))


with strategy.scope():
    predictions_dms = model_dms.predict(tf.data.Dataset.from_tensor_slices((test.sequence)).batch(128).map(lambda x: tokenizer(x)))

lens = test.sequence.str.len()

#
import numpy
predictions_a3_by_length = []
for i in range(lens.size):
    x = numpy.reshape(predictions_a3[i, :lens[i]], (-1, 1))
    x = numpy.clip(x, 0, 1)
    #x[:26] = 0
    #x[:-21] = 0
    predictions_a3_by_length.append(x)

predictions_a3 = numpy.concatenate(predictions_a3_by_length, 0)

#
import numpy
predictions_dms_by_length = []
for i in range(lens.size):
    x = numpy.reshape(predictions_dms[i, :lens[i]], (-1, 1))
    x = numpy.clip(x, 0, 1)
    #x[:26] = 0
    #x[:-21] = 0
    predictions_dms_by_length.append(x)

predictions_dms = numpy.concatenate(predictions_dms_by_length, 0)

predictions_dms.shape

#
from pandas import DataFrame
submission = DataFrame({'id':np.arange(0, 269796671, 1), 'reactivity_DMS_MaP': predictions_dms[:, 0], 'reactivity_2A3_MaP': predictions_a3[:, 0]})

submission.reactivity_2A3_MaP.describe()

submission.reactivity_DMS_MaP.describe()

#
!pip install pyarrow
submission.to_parquet('submission.parquet', index=False)



