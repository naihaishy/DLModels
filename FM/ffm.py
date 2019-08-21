# -*- coding:utf-8 -*-
# @Time : 2019/8/2 22:11
# @Author : naihai

from keras import Input, Model
from keras.layers import Layer, Dropout, Dense, multiply
import keras
import keras.backend as K
import tensorflow as tf


class FFMLayer(Layer):
    """
    自定义FFM层
    """

    def __init__(self, output_dim, factor_order, field_length, field_dict, activation=None, **kwargs):
        self.field_dict = field_dict
        self.field_length = field_length
        self.factor_order = factor_order
        self.output_dim = output_dim

        self.b = 0
        self.W = None
        self.V = None

        self.activation = keras.activations.get(activation)

        super(FFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]  # 输入数据的特征维度

        self.b = self.add_weight(name='bias', shape=(input_dim, self.output_dim),
                                 initializer='zero', trainable=True)
        self.W = self.add_weight(name='linear', shape=(input_dim, self.output_dim),
                                 initializer=keras.initializers.truncated_normal(0.0, 0.01), trainable=True)

        self.V = self.add_weight(name='interaction', shape=(input_dim, self.field_length, self.factor_order),
                                 initializer=keras.initializers.truncated_normal(0.0, 0.01), trainable=True)

        super(FFMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_dim = inputs.shape[1]  # 输入数据的特征维度

        field_cross = K.variable(0, dtype='float32')
        for i in range(input_dim):
            for j in range(i + 1, input_dim):
                a = K.dot(self.V[i, self.field_dict[j]], self.V[j, self.field_dict[i]])
                b = K.dot(inputs[:, i], inputs[:, j])

                field_cross = a * b

        interaction = field_cross
        output = self.b + K.dot(inputs, self.W) + interaction

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim


def ffm_model(feature_dim, lr, field_dict):
    inputs = Input(shape=(feature_dim,))
    fm_layer = FFMLayer(100, 30, 10, field_dict)(inputs)

    dropout_layer = Dropout(0.2)(fm_layer)
    outputs = Dense(1, activation='sigmoid')(dropout_layer)

    # 创建model
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def test():
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import train_test_split

    X, y = load_digits(n_class=2, return_X_y=True)
    y[y == 0] = -1
    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    field_dict = dict()
    for i in range(64):
        field_dict[i] = i % 10

    model = ffm_model(64, 0.1, field_dict)

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    pass


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    test()
