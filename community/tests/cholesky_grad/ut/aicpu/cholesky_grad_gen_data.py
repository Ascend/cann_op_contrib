"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op, dtypes

def write_file_txt_matrix(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')

def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()

def read_file_txt_to_boll(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)

def gen_data_file_spdmatrix(data_file, shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    rand_data_t = rand_data
    rand_data_t = rand_data_t.T
    data = np.matmul(rand_data_t,rand_data)
    write_file_txt_matrix(data_file, data, fmt="%s")
    return data

def gen_data_file_3d_spdmatrix(data_file, shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
        data = np.random.randint(low, high, size=shape)
    else:
        rand_data = np.random.uniform(low, high, size=shape)
        data = np.random.uniform(low, high, size=shape)
    for i in range(rand_data.shape[0]):
        data[i,:,:] = np.matmul(rand_data[i,:,:],rand_data[i,:,:].T)
    write_file_txt_matrix(data_file, data, fmt="%s")
    return data

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config

def gen_random_data_double():
    data_files=["choleskyGrad/data/choleskyGrad_data_input1_double.txt",
                "choleskyGrad/data/choleskyGrad_data_input2_double.txt",
                "choleskyGrad/data/choleskyGrad_data_output1_double.txt"]
    np.random.seed(23457)
    shape_x1 = [16, 16]
    A = gen_data_file_spdmatrix(data_files[0], shape_x1, np.float64, "uniform", -10, 10)
    L = gen_data_file_spdmatrix(data_files[1], shape_x1, np.float64, "uniform", -1, 1)

    x1 = tf.compat.v1.placeholder(tf.float64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float64, shape=shape_x1)
    x3 = tf.raw_ops.CholeskyGrad(l = x1,grad = x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(x3, feed_dict={x1:A, x2:L})
    write_file_txt_matrix(data_files[2], data, fmt="%s")

def gen_random_data_float():
    data_files=["choleskyGrad/data/choleskyGrad_data_input1_float.txt",
                "choleskyGrad/data/choleskyGrad_data_input2_float.txt",
                "choleskyGrad/data/choleskyGrad_data_output1_float.txt"]
    np.random.seed(23457)
    shape_x1 = [16, 16]
    A = gen_data_file_spdmatrix(data_files[0], shape_x1, np.float32, "uniform", -10, 10)
    L = gen_data_file_spdmatrix(data_files[1], shape_x1, np.float32, "uniform", -1, 1)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x3 = tf.raw_ops.CholeskyGrad(l = x1,grad = x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(x3, feed_dict={x1:A, x2:L})
    write_file_txt_matrix(data_files[2], data, fmt="%s")

def gen_random_data_float_big_data():
    data_files=["choleskyGrad/data/choleskyGrad_data_input1_float_1.txt",
                "choleskyGrad/data/choleskyGrad_data_input2_float_1.txt",
                "choleskyGrad/data/choleskyGrad_data_output1_float_1.txt"]
    np.random.seed(23457)
    shape_x1 = [512, 512]
    A = gen_data_file_spdmatrix(data_files[0], shape_x1, np.float32, "uniform", -10, 10)
    L = gen_data_file_spdmatrix(data_files[1], shape_x1, np.float32, "uniform", -1, 1)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x3 = tf.raw_ops.CholeskyGrad(l = x1,grad = x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(x3, feed_dict={x1:A, x2:L})
    write_file_txt_matrix(data_files[2], data, fmt="%s")

def gen_random_data_float_high_dim_1_data():
    data_files=["choleskyGrad/data/choleskyGrad_data_input1_3d_1.txt",
                "choleskyGrad/data/choleskyGrad_data_input2_3d_1.txt",
                "choleskyGrad/data/choleskyGrad_data_output1_3d_1.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 4, 4]
    A = gen_data_file_3d_spdmatrix(data_files[0], shape_x1, np.float32, "uniform", -10, 10)
    L = gen_data_file_3d_spdmatrix(data_files[1], shape_x1, np.float32, "uniform", -10, 10)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x3 = tf.raw_ops.CholeskyGrad(l = x1,grad = x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(x3, feed_dict={x1:A, x2:L})
    write_file_txt_matrix(data_files[2], data, fmt="%s")

def gen_random_data_float_high_dim_2_data():
    data_files=["choleskyGrad/data/choleskyGrad_data_input1_3d_2.txt",
                "choleskyGrad/data/choleskyGrad_data_input2_3d_2.txt",
                "choleskyGrad/data/choleskyGrad_data_output1_3d_2.txt"]
    np.random.seed(23457)
    shape_x1 = [32, 8, 8]
    A = gen_data_file_3d_spdmatrix(data_files[0], shape_x1, np.float32, "uniform", -10, 10)
    L = gen_data_file_3d_spdmatrix(data_files[1], shape_x1, np.float32, "uniform", -10, 10)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x3 = tf.raw_ops.CholeskyGrad(l = x1,grad = x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(x3, feed_dict={x1:A, x2:L})
    write_file_txt_matrix(data_files[2], data, fmt="%s")

def gen_random_data_float_high_dim_3_data():
    data_files=["choleskyGrad/data/choleskyGrad_data_input1_3d_3.txt",
                "choleskyGrad/data/choleskyGrad_data_input2_3d_3.txt",
                "choleskyGrad/data/choleskyGrad_data_output1_3d_3.txt"]
    np.random.seed(23457)
    shape_x1 = [128, 16, 16]
    A = gen_data_file_3d_spdmatrix(data_files[0], shape_x1, np.float32, "uniform", -10, 10)
    L = gen_data_file_3d_spdmatrix(data_files[1], shape_x1, np.float32, "uniform", -10, 10)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x3 = tf.raw_ops.CholeskyGrad(l = x1,grad = x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(x3, feed_dict={x1:A, x2:L})
    write_file_txt_matrix(data_files[2], data, fmt="%s")

def gen_random_data_float_unspd():
    data_files=["choleskyGrad/data/choleskyGrad_data_input1_float_unspd.txt",
                "choleskyGrad/data/choleskyGrad_data_input2_float_unspd.txt",
                "choleskyGrad/data/choleskyGrad_data_output1_float_unspd.txt"]
    shape_x1 = [2, 2]
    A = np.array([[1.0, 0.0],[1.0, 1.0]]).astype(np.float32)
    L = np.array([[2.0, 3.0],[3.0, 3.0]]).astype(np.float32)
    write_file_txt_matrix(data_files[0], A, fmt="%s") 
    write_file_txt_matrix(data_files[1], L, fmt="%s")

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x3 = tf.raw_ops.CholeskyGrad(l = x1,grad = x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(x3, feed_dict={x1:A, x2:L})
    write_file_txt_matrix(data_files[2], data, fmt="%s")

def run():
    gen_random_data_double()
    gen_random_data_float()
    gen_random_data_float_big_data()
    gen_random_data_float_high_dim_1_data()
    gen_random_data_float_high_dim_2_data()
    gen_random_data_float_high_dim_3_data()
    gen_random_data_float_unspd()


if __name__ == "__main__":
    run()