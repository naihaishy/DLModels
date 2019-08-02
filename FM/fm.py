# -*- coding:utf-8 -*-
# @Time : 2019/8/2 19:43
# @Author : naihai

import keras
from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.layers import Layer
import keras.backend as K


class FMLayer(Layer):
    """
    自定义FM层
    """

    def __init__(self, input_dim, output_dim=30, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel = None
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        定义权重 为该层创建一个可训练的权重
        :param input_shape:
        :return:
        """
        self.kernel = self.add_weight(name="kernel", shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_normal', trainable=True)
        super(FMLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs, **kwargs):
        """
        该层的逻辑功能
        :param inputs:
        :param kwargs:
        :return:
        """
        a = K.pow(K.dot(inputs, self.kernel), 2)
        b = K.dot(K.pow(inputs, 2), K.pow(self.kernel, 2))
        return K.mean(a - b, 1, keepdims=True) * 0.5

    def compute_output_shape(self, input_shape):
        """
        如果你的层更改了输入张量的形状
        在这里定义形状变化的逻辑
        :param input_shape:
        :return:
        """
        return input_shape[0], self.output_dim


def FMModel(feature_dim, num_class, lr):
    inputs = Input(shape=(feature_dim,))
    # 线性层
    linear_layer = Dense(units=1,
                         bias_regularizer=keras.regularizers.l2(0.01),
                         kernel_regularizer=keras.regularizers.l1(0.01))(inputs)
    # 交互层
    cross_layer = FMLayer(input_dim=feature_dim)(inputs)

    # 将线性层与交互层合并
    fm_layer = keras.layers.Add()([linear_layer, cross_layer])

    # 为fm layer添加激活函数
    predictions = keras.layers.Activation('sigmoid')(fm_layer)

    # 创建model
    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=keras.optimizers.SGD(lr),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    return model


def test():
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import scale, normalize
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X, y = load_digits(n_class=2, return_X_y=True)
    y[y == 0] = -1
    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)
    model = FMModel(64, 10, 0.1)

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    pres = model.predict(X_test)
    print(accuracy_score(y_test, pres))


if __name__ == '__main__':
    test()

    pass
