# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op, dtypes

# prama1: file_name: the file which store the data
# param2: data: data which will be stored
# param3: fmt: format
def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')

# prama1: file_name: the file which store the data
# param2: dtype: data type
# param3: delim: delimiter which is used to split data
def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()

# prama1: file_name: the file which store the data
# param2: delim: delimiter which is used to split data
def read_file_txt_to_bool(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(Falsetf)
        else:
            bool_data.append(True)
    return np.array(bool_data)

# prama1: data_file: the file which store the generation data
# param2: shape: data shape
# param3: dtype: data type
# param4: rand_type: the method of generate data, select from "randint, uniform"
# param5: data lower limit
# param6: data upper limit
def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    elif rand_type == "uniform":
        rand_data = np.random.uniform(low, high, size=shape)
    else:
       r1=np.random.uniform(low,high,size=shape)
       rand_data=np.empty((shape[0],shape[1],shape[2]),dtype=dtype)
       for i in range(0,shape[0]):
        for p in range(0,shape[1]):
            for k in range(0,shape[2]):  
                rand_data[i,p,k]=complex(r1[i,p,k])
    if dtype == np.float16:
        data = np.array(rand_data, dtype=np.float32)
        write_file_txt(data_file, data, fmt="%s")
        data = np.array(data, dtype=np.float16)
    else:
        data = np.array(rand_data, dtype=dtype)
        write_file_txt(data_file, data, fmt="%s")
    return data

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config

def gen_random_data_float1():
    data_files=["tanh/data/tanh_data_input1_1.txt",               
                "tanh/data/tanh_data_output1_1.txt"]
    np.random.seed(23457)
    shape_x = [15]
    
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    
    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    
    re = tf.raw_ops.Tanh(x=x,name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data, fmt="%s")
def gen_random_data_float2():
    data_files=["tanh/data/tanh_data_input1_2.txt",               
                "tanh/data/tanh_data_output1_2.txt"]
    np.random.seed(23457)
    shape_x = [1250, 1024]
    
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    
    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    
    re = tf.raw_ops.Tanh(x=x,name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_float3():
    data_files=["tanh/data/tanh_data_input1_3.txt",               
                "tanh/data/tanh_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x = [15, 1025, 30]
    
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    
    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    
    re = tf.raw_ops.Tanh(x=x,name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_double1():
    data_files=["tanh/data/tanh_data_input1_4.txt",
                "tanh/data/tanh_data_output1_4.txt"]
    np.random.seed(3457)
    shape_x = [1170]
    a = gen_data_file(data_files[0], shape_x, np.float64, "uniform", -100, 100)

    x = tf.compat.v1.placeholder(tf.float64, shape=shape_x)

    re = tf.raw_ops.Tanh(x=x,name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_double2():
    data_files=["tanh/data/tanh_data_input1_5.txt",
                "tanh/data/tanh_data_output1_5.txt"]
    np.random.seed(3457)
    shape_x = [7, 12]
    a = gen_data_file(data_files[0], shape_x, np.float64, "uniform", -100, 100)

    x = tf.compat.v1.placeholder(tf.float64, shape=shape_x)

    re = tf.raw_ops.Tanh(x=x,name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_double3():
    data_files=["tanh/data/tanh_data_input1_6.txt",
                "tanh/data/tanh_data_output1_6.txt"]
    np.random.seed(3457)
    shape_x = [7, 1200, 30]
    a = gen_data_file(data_files[0], shape_x, np.float64, "uniform", -100, 100)

    x = tf.compat.v1.placeholder(tf.float64, shape=shape_x)

    re = tf.raw_ops.Tanh(x=x,name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_float16():
    data_files=["tanh/data/tanh_data_input1_7.txt",
               
                "tanh/data/tanh_data_output1_7.txt"]
    np.random.seed(3457)
    shape_x = [256,1024]
    a = gen_data_file(data_files[0], shape_x, np.float16, "uniform", -10, 10)

    x = tf.compat.v1.placeholder(tf.float16, shape=shape_x)

    re = tf.raw_ops.Tanh(x=x,name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex64():
    data_files=["tanh/data/tanh_data_input1_8.txt",

                "tanh/data/tanh_data_output1_8.txt"]
    np.random.seed(3457)
    shape_x = [1300, 10, 4]

    a = gen_data_file(data_files[0], shape_x, np.complex64, "complex",  0, 100)

    x = tf.compat.v1.placeholder(tf.complex64, shape=shape_x)

    re = tf.raw_ops.Tanh(x=x,name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data, fmt="%s")
  
def gen_random_data_complex128():
    data_files=["tanh/data/tanh_data_input1_9.txt",
                "tanh/data/tanh_data_output1_9.txt"]
    np.random.seed(677)
    shape_x = [6,6,6]

    a = gen_data_file(data_files[0], shape_x, np.complex128,"complex",  0, 100)

    x = tf.compat.v1.placeholder(tf.complex128, shape=shape_x)
   
    re = tf.raw_ops.Tanh(x=x,name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data, fmt="%s")
  
def run():

    gen_random_data_float1()
    gen_random_data_float2()
    gen_random_data_float3()
    gen_random_data_double1()
    gen_random_data_double2()
    gen_random_data_double3()
    gen_random_data_float16()
    gen_random_data_complex64()
    gen_random_data_complex128()


if __name__ == "__main__":
    run()