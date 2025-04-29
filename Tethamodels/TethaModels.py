import tensorflow as tf 
import keras 
import numpy as np 


@keras.saving.register_keras_serializable()
def positionalEncoding (d_model,position) :
    position = np.arange(position)[:,np.newaxis]
    dim = np.arange(d_model)[np.newaxis,:]
    div_values = np.power(10000,(2* dim //2)) / tf.cast(d_model,dtype=tf.float32)
    div_values= np.maximum(div_values,1e-9)
    angle_rads = position * div_values
    angle_rads[:,0::2] = np.sin(angle_rads[:,0::2])
    angle_rads[:,1::2] = np.cos(angle_rads[:,1::2])
    return tf.cast(angle_rads,dtype=tf.float32)

@keras.saving.register_keras_serializable()
class SelfAttention(keras.layers.Layer):
    def __init__(self, key_dim):
        super(SelfAttention, self).__init__()
        self.key_dim = key_dim

    def build(self, input_shape):
        features = input_shape[-1]
        self.Wq = self.add_weight(shape=(features, self.key_dim), trainable=True, initializer="random_normal")
        self.Wk = self.add_weight(shape=(features, self.key_dim), trainable=True, initializer="random_normal")
        self.Wv = self.add_weight(shape=(features, self.key_dim), trainable=True, initializer="random_normal")

    def call(self, x):
        q = tf.matmul(x, self.Wq)
        k = tf.matmul(x, self.Wk)
        v = tf.matmul(x, self.Wv)
        key_dim = tf.cast(self.key_dim, tf.float32)
        scores = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])) / tf.math.sqrt(key_dim)
        weights = tf.nn.softmax(scores, axis=-1)
        attention = tf.matmul(weights, v)
        return attention

    def get_config(self):
        config = super().get_config()
        config.update({
            "key_dim": self.key_dim
        })
        return config

@keras.saving.register_keras_serializable()
class BlockEncoder(keras.layers.Layer):
    def __init__(self, d_model, ffn, drop_rate=0.1, epsilon=1e-6):
        super(BlockEncoder, self).__init__()
        self.d_model = d_model
        self.ffn = ffn
        self.drop_rate = drop_rate
        self.epsilon = epsilon

        self.Attention = SelfAttention(d_model)
        self.Feed_forward_nn = keras.Sequential([
            keras.layers.Dense(ffn, activation=keras.activations.swish),
            keras.layers.Dense(d_model)
        ])
        self.Normal1 = keras.layers.LayerNormalization(epsilon=epsilon)
        self.Normal2 = keras.layers.LayerNormalization(epsilon=epsilon)
        self.dropout1 = keras.layers.Dropout(drop_rate)
        self.dropout2 = keras.layers.Dropout(drop_rate)

    def call(self, x):
        attn = self.Attention(x)
        attn = self.dropout1(attn)
        attn = self.Normal1(attn + x)
        ffn = self.Feed_forward_nn(attn)
        ffn = self.dropout2(ffn)
        ffn = self.Normal2(ffn + attn)
        return ffn

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "ffn": self.ffn,
            "drop_rate": self.drop_rate,
            "epsilon": self.epsilon
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
@keras.saving.register_keras_serializable()
class Tetha(keras.Model):
    def __init__(self, vocab_Size, drop_rate, num_layers=32, epsilon=1e-6, maxpos=100):
        super(Tetha, self).__init__()
        self.vocab_Size = vocab_Size
        self.drop_rate = drop_rate
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.maxpos = maxpos
        
        self.Embedding = keras.layers.Embedding(vocab_Size, 16)
        self.BlockEncod = [BlockEncoder(16, 32, drop_rate=drop_rate, epsilon=epsilon) for _ in range(num_layers)]
        self.PosEncoding = self.add_weight(
            shape=positionalEncoding(16, maxpos).shape,
                    initializer=tf.constant_initializer(positionalEncoding(16, maxpos).numpy()),
                    trainable=False,
                    name="positional_encoding"
                )

    
    def call(self, x):
        seq_len = x.shape[1]
        x = self.Embedding(x)
        x += self.PosEncoding[:seq_len, :]
        for block in self.BlockEncod:
            x = block(x)
        return x

    def get_config(self):
        return {
            "vocab_Size": self.vocab_Size,
            "drop_rate": self.drop_rate,
            "num_layers": self.num_layers,
            "epsilon": self.epsilon,
            "maxpos": self.maxpos
        }

