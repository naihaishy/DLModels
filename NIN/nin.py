# -*- coding:utf-8 -*-
# @Time : 2019/7/24 11:21
# @Author : naihai

from keras import Model
from keras.layers import Input, Dense

if __name__ == '__main__':
    a = Input(shape=(32,))
    b = Dense(32)(a)

    model = Model()
