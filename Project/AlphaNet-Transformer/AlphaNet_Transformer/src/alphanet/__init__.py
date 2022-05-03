import tensorflow as _tf
import tensorflow.keras.layers as _tfl
from tensorflow.keras.layers import Layer as _Layer
from tensorflow.keras.initializers import Initializer as _Initializer
from tensorflow.keras import Model as _Model
from tensorflow.keras import Sequential as _Sequential
from .metrics import UpDownAccuracy as _UpDownAccuracy
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

if not "2.3.0" <= _tf.__version__:
    print(f"requires tensorflow version >= 2.3.0, "
          f"current version {_tf.__version__}")
    exit(1)

__all__ = ["Std",
           "Return",
           "Correlation",
           "LinearDecay",
           "Covariance",
           "ZScore",
           "FeatureExpansion",
           "TransformerBlock",
           "TokenAndPositionEmbedding",
           "AlphaNetV2",
           "AlphaNetV3",
           "AlphaNetV4",
           "load_model"]


class _StrideLayer(_Layer, _ABC):

    def __init__(self, stride=10, **kwargs):
        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(_StrideLayer, self).__init__(**kwargs)
        self.stride = stride
        self.out_shape = None
        self.intermediate_shape = None

    def build(self, input_shape):
        (features,
         output_length) = __get_dimensions__(input_shape, self.stride)
        self.out_shape = [-1, output_length, features]
        self.intermediate_shape = [-1, self.stride, features]

    def get_config(self):
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config


class Max(_StrideLayer):
    def call(self, inputs, *args, **kwargs):
        strides = _tf.reshape(inputs, self.intermediate_shape)

        # compute max for each stride
        std = _tf.math.reduce_max(strides, axis=-2)
        return _tf.reshape(std, self.out_shape)

class Min(_StrideLayer):
    def call(self, inputs, *args, **kwargs):
        strides = _tf.reshape(inputs, self.intermediate_shape)

        # compute max for each stride
        std = _tf.math.reduce_min(strides, axis=-2)
        return _tf.reshape(std, self.out_shape)
    
    
class Std(_StrideLayer):
    def call(self, inputs, *args, **kwargs):
        strides = _tf.reshape(inputs, self.intermediate_shape)

        # compute standard deviations for each stride
        std = _tf.math.reduce_std(strides, axis=-2)
        return _tf.reshape(std, self.out_shape)


class ZScore(_StrideLayer):
    def call(self, inputs, *args, **kwargs):
        strides = _tf.reshape(inputs, self.intermediate_shape)

        # compute standard deviations for each stride
        std = _tf.math.reduce_std(strides, axis=-2)

        # compute means for each stride
        means = _tf.math.reduce_mean(strides, axis=-2)

        # divide means by standard deviations for each stride
        z_score = _tf.math.divide_no_nan(means, std)
        return _tf.reshape(z_score, self.out_shape)


class LinearDecay(_StrideLayer):
    def call(self, inputs, *args, **kwargs):
        # get linear decay kernel
        single_kernel = _tf.linspace(1.0, self.stride, num=self.stride)
        kernel = _tf.repeat(single_kernel, self.intermediate_shape[2])
        kernel = kernel / _tf.reduce_sum(single_kernel)

        # reshape tensors into:
        # (bash_size * (time_steps / stride), stride, features)
        kernel = _tf.reshape(kernel, self.intermediate_shape[1:])
        inputs = _tf.reshape(inputs, self.intermediate_shape)

        # broadcasting kernel to inputs batch dimension
        linear_decay = _tf.reduce_sum(kernel * inputs, axis=1)
        linear_decay = _tf.reshape(linear_decay, self.out_shape)
        return linear_decay


class Return(_Layer):
    def __init__(self, stride=10, **kwargs):
        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(Return, self).__init__(**kwargs)
        self.stride = stride

    def build(self, input_shape):
        time_steps = input_shape[1]
        if time_steps % self.stride != 0:
            raise ValueError("Error, time_steps should be n * stride")

    def call(self, inputs, *args, **kwargs):
        # get the endings of each strides as numerators
        numerators = inputs[:, (self.stride - 1)::self.stride, :]

        # get the beginnings of each strides as denominators
        denominators = inputs[:, 0::self.stride, :]

        return _tf.math.divide_no_nan(numerators, denominators) - 1.0

    def get_config(self):
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config


class _OuterProductLayer(_Layer, _ABC):

    def __init__(self, stride=10, **kwargs):
        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(_OuterProductLayer, self).__init__(**kwargs)
        self.stride = stride
        self.intermediate_shape = None
        self.out_shape = None
        self.lower_mask = None

    def build(self, input_shape):
        (features,
         output_length) = __get_dimensions__(input_shape, self.stride)
        self.intermediate_shape = (-1, self.stride, features)
        output_features = int(features * (features - 1) / 2)
        self.out_shape = (-1, output_length, output_features)
        self.lower_mask = _LowerNoDiagonalMask()((features, features))

    def get_config(self):
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config

    @_abstractmethod
    def call(self, inputs, *args, **kwargs):
        ...


class Covariance(_OuterProductLayer):
    def call(self, inputs, *args, **kwargs):
        # compute means for each stride
        means = _tf.nn.avg_pool(inputs,
                                ksize=self.stride,
                                strides=self.stride,
                                padding="VALID")

        # subtract means for each stride
        means_broadcast = _tf.repeat(means, self.stride, axis=1)
        means_subtracted = _tf.subtract(inputs, means_broadcast)
        means_subtracted = _tf.reshape(means_subtracted,
                                       self.intermediate_shape)

        # compute covariance matrix
        covariance_matrix = _tf.einsum("ijk,ijm->ikm",
                                       means_subtracted,
                                       means_subtracted)
        covariance_matrix = covariance_matrix / (self.stride - 1)

        # get the lower part of the covariance matrix
        # without the diagonal elements
        covariances = _tf.boolean_mask(covariance_matrix,
                                       self.lower_mask,
                                       axis=1)
        covariances = _tf.reshape(covariances, self.out_shape)
        return covariances


class Correlation(_OuterProductLayer):
    def call(self, inputs, *args, **kwargs):
        # compute means for each stride
        means = _tf.nn.avg_pool(inputs,
                                ksize=self.stride,
                                strides=self.stride,
                                padding="VALID")

        # subtract means for each stride
        means_broadcast = _tf.repeat(means, self.stride, axis=1)
        means_subtracted = _tf.subtract(inputs, means_broadcast)
        means_subtracted = _tf.reshape(means_subtracted,
                                       self.intermediate_shape)

        # compute standard deviations for each strides
        squared_diff = _tf.square(means_subtracted)
        mean_squared_error = _tf.reduce_mean(squared_diff, axis=1)
        std = _tf.sqrt(mean_squared_error)

        # get denominator of correlation matrix
        denominator_matrix = _tf.einsum("ik,im->ikm", std, std)

        # compute covariance matrix
        covariance_matrix = _tf.einsum("ijk,ijm->ikm",
                                       means_subtracted,
                                       means_subtracted)
        covariance_matrix = covariance_matrix / self.stride

        # take the lower triangle of each matrix without diagonal
        covariances = _tf.boolean_mask(covariance_matrix,
                                       self.lower_mask,
                                       axis=1)
        denominators = _tf.boolean_mask(denominator_matrix,
                                        self.lower_mask,
                                        axis=1)
        correlations = _tf.math.divide_no_nan(covariances, denominators)
        correlations = _tf.reshape(correlations, self.out_shape)
        return correlations


class FeatureExpansion(_Layer):
    def __init__(self, stride=10, **kwargs):
        if type(stride) is not int or stride <= 1:
            raise ValueError("Illegal Argument: stride should be an integer "
                             "greater than 1")
        super(FeatureExpansion, self).__init__(**kwargs)
        self.stride = stride
        self.std = _tf.function(Std(stride=self.stride))
        self.z_score = _tf.function(ZScore(stride=self.stride))
        self.linear_decay = _tf.function(LinearDecay(stride=self.stride))
        self.return_ = _tf.function(Return(stride=self.stride))
        self.covariance = _tf.function(Covariance(stride=self.stride))
        self.correlation = _tf.function(Correlation(stride=self.stride))
        
        self.max = _tf.function(Max(stride=self.stride))
        self.min = _tf.function(Min(stride=self.stride))

    def call(self, inputs, *args, **kwargs):
        std_output = self.std(inputs)
        z_score_output = self.z_score(inputs)
        decay_linear_output = self.linear_decay(inputs)
        return_output = self.return_(inputs)
        covariance_output = self.covariance(inputs)
        correlation_output = self.correlation(inputs)
        
        max_output = self.max(inputs)
        min_output = self.min(inputs)
        
        return _tf.concat([std_output,
                           z_score_output,
                           decay_linear_output,
                           return_output,
                           covariance_output,
                           correlation_output], axis=2)
        # return _tf.concat([std_output,
        #                    z_score_output,
        #                    decay_linear_output,
        #                    return_output,
        #                    covariance_output,
        #                    correlation_output,
        #                    max_output,
        #                    min_output], axis=2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config


class TransformerBlock(_Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = _tfl.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # self.ffn = _Sequential(
        #     [_tfl.Dense(ff_dim, activation="relu"), _tfl.Dense(embed_dim),]
        # )
        self.ffn1 = _tfl.Dense(ff_dim, activation="relu")
        self.ffn2 = _tfl.Dense(embed_dim)
        self.layernorm1 = _tfl.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = _tfl.LayerNormalization(epsilon=1e-6)
        self.dropout1 = _tfl.Dropout(rate)
        self.dropout2 = _tfl.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn1(out1)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'embed_dim': self.embed_dim,
                       'num_heads': self.num_heads,
                       'ff_dim': self.ff_dim,
                       'rate': self.rate})
        return config

class TokenAndPositionEmbedding(_Layer):
    def __init__(self, maxlen, embed_dim, time_num):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.time_num = time_num
        # self.token_emb = _tfl.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = _tfl.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.pos_range = _tf.range(start=0, limit=time_num, delta=1)

    def call(self, x):
        # maxlen = _tf.shape(x)[-2]
        positions = self.pos_emb(self.pos_range)
        # print(positions)
        # print(x)
        # x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'maxlen': self.maxlen,
                       'embed_dim': self.embed_dim,
                       'time_num': self.time_num})
        return config
    

class AlphaNet(_Model):
    def __init__(self,
                 dropout=0.0,
                 l2=0.001,
                 classification=False,
                 categories=0,
                 recurrent_unit="GRU",
                 hidden_units=30,
                 *args,
                 **kwargs):
        super(AlphaNet, self).__init__(*args, **kwargs)
        self.l2 = l2
        self.dropout = dropout
        self.expanded10 = FeatureExpansion(stride=10)
        self.expanded5 = FeatureExpansion(stride=5)
        # self.expanded3 = FeatureExpansion(stride=3)
        self.normalized10 = _tfl.BatchNormalization()
        self.normalized5 = _tfl.BatchNormalization()
        # self.normalized10_1 = _tfl.BatchNormalization()
        # self.normalized5_1 = _tfl.BatchNormalization()
        # self.normalized3 = _tfl.BatchNormalization()
        self.dropout_layer = _tfl.Dropout(self.dropout)
        if recurrent_unit == "GRU":
            self.recurrent10 = _tfl.GRU(units=hidden_units)
            self.recurrent5 = _tfl.GRU(units=hidden_units)
        elif recurrent_unit == "LSTM":
            self.recurrent10 = _tfl.LSTM(units=hidden_units)
            self.recurrent5 = _tfl.LSTM(units=hidden_units)
        elif recurrent_unit == "Transformer":
            # self.posembed3_1 = TokenAndPositionEmbedding(10, 88, 10)
            # self.transformer3_1 = TransformerBlock(88, 4, 88, self.dropout)
            self.posembed5_1 = TokenAndPositionEmbedding(10, 88, 6)
            self.transformer5_1 = TransformerBlock(88, 4, 88, self.dropout)
            self.posembed10_1 = TokenAndPositionEmbedding(10, 88, 3)
            self.transformer10_1 = TransformerBlock(88, 4, 88, self.dropout)
            
            # self.posembed3_2 = TokenAndPositionEmbedding(10, 88, 10)
            # self.transformer3_2 = TransformerBlock(88, 2, 88, self.dropout)
            self.posembed5_2 = TokenAndPositionEmbedding(10, 88, 6)
            self.transformer5_2 = TransformerBlock(88, 2, 88, self.dropout)
            self.posembed10_2 = TokenAndPositionEmbedding(10, 88, 3)
            self.transformer10_2 = TransformerBlock(88, 2, 88, self.dropout)
            
            # self.posembed5_3 = TokenAndPositionEmbedding(10, 88, 6)
            # self.transformer5_3 = TransformerBlock(88, 4, 88, self.dropout)
            # self.posembed10_3 = TokenAndPositionEmbedding(10, 88, 3)
            # self.transformer10_3 = TransformerBlock(88, 4, 88, self.dropout)
            # self.att = _tfl.MultiHeadAttention(num_heads=2, key_dim=32)
        else:
            raise ValueError("Unknown recurrent_unit")
        self.normalized10_2 = _tfl.BatchNormalization()
        self.normalized5_2 = _tfl.BatchNormalization()
        # self.normalized3_2 = _tfl.BatchNormalization()
        self.reshape10 = _tfl.Reshape([264])
        self.reshape5 = _tfl.Reshape([528])
        # self.reshape3 = _tfl.Reshape([880])
        self.concat1 = _tfl.Concatenate(axis=-1)
        # self.concat2 = _tfl.Concatenate(axis=-1)
        self.regularizer = _tf.keras.regularizers.l2(self.l2)
        if classification:
            if categories < 1:
                raise ValueError("categories should be at least 1")
            elif categories == 1:
                self.outputs = _tfl.Dense(1, activation="sigmoid",
                                          kernel_initializer="truncated_normal",
                                          kernel_regularizer=self.regularizer)
            else:
                self.outputs = _tfl.Dense(categories, activation="softmax",
                                          kernel_initializer="truncated_normal",
                                          kernel_regularizer=self.regularizer)
        else:
            # self.outputs1 = _tfl.Dense(264, activation="tanh",
            #                           kernel_initializer="truncated_normal",
            #                           kernel_regularizer=self.regularizer)
            # self.outputs2 = _tfl.Dense(1, activation="tanh",
            #                           kernel_initializer="truncated_normal",
            #                           kernel_regularizer=self.regularizer)
            self.outputs = _tfl.Dense(1, activation="linear",
                                      kernel_initializer="truncated_normal",
                                      kernel_regularizer=self.regularizer)

    @_tf.function
    def call(self, inputs, training=None, mask=None):
        expanded10 = self.expanded10(inputs)
        expanded5 = self.expanded5(inputs)
        # expanded3 = self.expanded3(inputs)
        normalized10 = self.normalized10(expanded10, training=training)
        normalized5 = self.normalized5(expanded5, training=training)
        # normalized3 = self.normalized3(expanded3, training=training)
        
        posembed10_1 = self.posembed10_1(normalized10)
        recurrent10_1 = self.transformer10_1(posembed10_1)
        posembed5_1 = self.posembed5_1(normalized5)
        recurrent5_1 = self.transformer5_1(posembed5_1)
        # posembed3_1 = self.posembed3_1(normalized3)
        # recurrent3_1 = self.transformer3_1(posembed3_1)
        
        # normalized10_1 = self.normalized10_1(recurrent10_1, training=training)
        # normalized5_1 = self.normalized5_1(recurrent5_1, training=training)
    
        posembed10_2 = self.posembed10_2(recurrent10_1)
        recurrent10_2 = self.transformer10_2(posembed10_2)
        posembed5_2 = self.posembed5_2(recurrent5_1)
        recurrent5_2 = self.transformer5_2(posembed5_2)
        # posembed3_2 = self.posembed3_2(recurrent3_1)
        # recurrent3_2 = self.transformer3_2(posembed3_2)
        
        # posembed10_3 = self.posembed10_3(recurrent10_2)
        # recurrent10_3 = self.transformer10_3(posembed10_3)
        # posembed5_3 = self.posembed5_3(recurrent5_2)
        # recurrent5_3 = self.transformer5_3(posembed5_3)
        
        # pool10 = _tfl.GlobalMaxPooling1D()(recurrent10)
        # pool5 = _tfl.GlobalMaxPooling1D()(recurrent5)
        # pool3 = _tfl.GlobalMaxPooling1D()(recurrent3)
        # pool10 = recurrent10
        # pool5 = recurrent5
        # pool3 = recurrent3
        # recurrent10 = self.transformer(posembed10)
        
        # posembed5 = self.posembed(normalized5)
        # recurrent5 = self.transformer(posembed5)
        
        normalized10_2 = self.normalized10_2(recurrent10_2, training=training)
        normalized5_2 = self.normalized5_2(recurrent5_2, training=training)
        # normalized3_2 = self.normalized3_2(recurrent3_2, training=training)
        # batch_size = _tf.shape(normalized10_2)[0]
        normalized10_2_flat = self.reshape10(normalized10_2)
        normalized5_2_flat = self.reshape5(normalized5_2)
        # normalized3_2_flat = self.reshape3(normalized3_2)
        # normalized10_2 = pool10
        # normalized5_2 = pool5
        concat = self.concat1([normalized10_2_flat, normalized5_2_flat])
        # concat = self.concat2([concat, normalized3_2_flat])
        dropout = self.dropout_layer(concat, training=training)
        # output1 = self.outputs1(dropout)
        # output2 = self.outputs2(output1)
        output = self.outputs(dropout)
        return output

    def compile(self,
                optimizer=_tf.keras.optimizers.Adam(0.0001),
                loss="MSE",
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'dropout': self.dropout,
                       'l2': self.l2})
        return config


class AlphaNetV3(_Model):
    def __init__(self,
                 dropout=0.0,
                 l2=0.001,
                 classification=False,
                 categories=0,
                 recurrent_unit="GRU",
                 hidden_units=30,
                 *args,
                 **kwargs):
        super(AlphaNetV3, self).__init__(*args, **kwargs)
        self.l2 = l2
        self.dropout = dropout
        self.expanded10 = FeatureExpansion(stride=10)
        self.expanded5 = FeatureExpansion(stride=5)
        self.normalized10 = _tfl.BatchNormalization()
        self.normalized5 = _tfl.BatchNormalization()
        self.dropout_layer = _tfl.Dropout(self.dropout)
        if recurrent_unit == "GRU":
            self.recurrent10 = _tfl.GRU(units=hidden_units)
            self.recurrent5 = _tfl.GRU(units=hidden_units)
        elif recurrent_unit == "LSTM":
            self.recurrent10 = _tfl.LSTM(units=hidden_units)
            self.recurrent5 = _tfl.LSTM(units=hidden_units)
        else:
            raise ValueError("Unknown recurrent_unit")
        self.normalized10_2 = _tfl.BatchNormalization()
        self.normalized5_2 = _tfl.BatchNormalization()
        self.concat = _tfl.Concatenate(axis=-1)
        self.regularizer = _tf.keras.regularizers.l2(self.l2)
        if classification:
            if categories < 1:
                raise ValueError("categories should be at least 1")
            elif categories == 1:
                self.outputs = _tfl.Dense(1, activation="sigmoid",
                                          kernel_initializer="truncated_normal",
                                          kernel_regularizer=self.regularizer)
            else:
                self.outputs = _tfl.Dense(categories, activation="softmax",
                                          kernel_initializer="truncated_normal",
                                          kernel_regularizer=self.regularizer)
        else:
            self.outputs = _tfl.Dense(1, activation="linear",
                                      kernel_initializer="truncated_normal",
                                      kernel_regularizer=self.regularizer)

    @_tf.function
    def call(self, inputs, training=None, mask=None):
        expanded10 = self.expanded10(inputs)
        expanded5 = self.expanded5(inputs)
        normalized10 = self.normalized10(expanded10, training=training)
        normalized5 = self.normalized5(expanded5, training=training)
        recurrent10 = self.recurrent10(normalized10)
        recurrent5 = self.recurrent5(normalized5)
        normalized10_2 = self.normalized10_2(recurrent10, training=training)
        normalized5_2 = self.normalized5_2(recurrent5, training=training)
        concat = self.concat([normalized10_2, normalized5_2])
        dropout = self.dropout_layer(concat, training=training)
        output = self.outputs(dropout)
        return output

    def compile(self,
                optimizer=_tf.keras.optimizers.Adam(0.0001),
                loss="MSE",
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'dropout': self.dropout,
                       'l2': self.l2})
        return config



def load_model(filepath,
               custom_objects: dict = None,
               compile: bool = True,
               options=None):

    object_dict = {"UpDownAccuracy": _UpDownAccuracy}
    if custom_objects is not None:
        object_dict.update(custom_objects)
    return _tf.keras.models.load_model(filepath,
                                       custom_objects=object_dict,
                                       compile=compile,
                                       options=options)


class _LowerNoDiagonalMask(_Initializer):
    def __init__(self):
        super(_LowerNoDiagonalMask, self).__init__()

    def __call__(self, shape, **kwargs):
        ones = _tf.ones(shape)
        mask_lower = _tf.linalg.band_part(ones, -1, 0)
        mask_diag = _tf.linalg.band_part(ones, 0, 0)
        # lower triangle removing the diagonal elements
        mask = _tf.cast(mask_lower - mask_diag, dtype=_tf.bool)
        return mask


def __get_dimensions__(input_shape, stride):
    if type(stride) is not int or stride <= 1:
        raise ValueError("Illegal Argument: stride should be an integer "
                         "greater than 1")
    time_steps = input_shape[1]
    features = input_shape[2]
    output_length = time_steps // stride

    if time_steps % stride != 0:
        raise ValueError("Error, time_steps should be n * stride")

    return features, output_length
