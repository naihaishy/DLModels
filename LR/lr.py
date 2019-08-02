# -*- coding:utf-8 -*-
# @Time : 2019/8/2 19:48
# @Author : naihai

"""
就是一个Dense层
"""
from keras.layers import Dense, Input
import keras


def LogisticRegressionModel(input_dim, units, weight_reg, bias_reg, lr):
    inputs = Input(shape=(input_dim,))
    outputs = Dense(units=units, use_bias=True,
                    bias_regularizer=keras.regularizers.l2(bias_reg),
                    kernel_regularizer=keras.regularizers.l1(weight_reg),
                    activation=keras.activations.sigmoid)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.SGD(lr),
                  loss=keras.losses.binary_crossentropy,
                  metrics=keras.metrics.binary_accuracy)
    return model


if __name__ == '__main__':
    lr = LogisticRegressionModel(64, 2, 0.01, 0.01, 0.1)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    lr.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test))
