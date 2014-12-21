#!/usr/bin/env python2
#-*- coding: UTF-8 -*-
#File:
#Date:
#Author: Yang Liu <largelymfs@gmail.com>
#Description:
with open("para_vectors_train.txt") as f:
    with open("para_train.txt","w") as fo:
        for i in range(100000):
            f.readline()
        for i in range(25000):
            f.readline()
            fo.write(f.readline())

