# -*- coding:utf-8 -*-
# @Time : 2019/8/2 19:43
# @Author : naihai

import keras
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.layers import Layer
from keras.models import Model
from keras.utils import plot_model

class FMLayer(Layer):
    """
    自定义FM层
    """

    def __init__(self, output_dim, factor_order, activation=None, **kwargs):
        self.factor_order = factor_order
        self.output_dim = output_dim

        self.b = 0
        self.W = None
        self.V = None

        self.activation = keras.activations.get(activation)

        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        定义权重 为该层创建一个可训练的权重
        :param input_shape: 为2元组 输入数据的shape batch_size feature_length
        :return:
        """
        assert len(input_shape) == 2
        input_dim = input_shape[1]  # 输入数据的特征维度

        self.b = self.add_weight(name='bias', shape=(self.output_dim,),
                                 initializer='zeros', trainable=True)

        self.W = self.add_weight('linear', shape=(input_dim, self.output_dim),
                                 initializer='glorot_normal', trainable=True)

        self.V = self.add_weight('interaction', shape=(input_dim, self.factor_order),
                                 initializer='glorot_normal', trainable=True)

        super(FMLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs, **kwargs):
        """
        该层的逻辑功能
        :param inputs:
        :param kwargs:
        :return:
        """
        a = K.pow(K.dot(inputs, self.V), 2)
        b = K.dot(K.pow(inputs, 2), K.pow(self.V, 2))
        interaction = K.sum(a - b, 1, keepdims=True) * 0.5

        output = self.b + K.dot(inputs, self.W) + interaction

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        """
        如果你的层更改了输入张量的形状
        在这里定义形状变化的逻辑
        :param input_shape:
        :return:
        """
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim


def fm_model(feature_dim, lr):
    inputs = Input(shape=(feature_dim,))
    fm_layer = FMLayer(100, 30)(inputs)

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
    model = fm_model(64, 0.1)

    # plot_model(model, 'model.png', show_shapes=True)

    model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    test()
