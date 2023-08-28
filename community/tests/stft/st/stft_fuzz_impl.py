# Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

import numpy as np
import torch
import random
random.seed(17)
np.random.seed(17)


def calc_expect_func(x, y, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided):

    x_val = x["value"]
    if onesided:
        x_re = np.array(x_val[:,:, 0])        
        torch_data = torch.from_numpy(x_re)
    else:
        x_re = np.array(x_val[:,:, 0])
        x_im = np.array(x_val[:,:, 1])
        
        real_data = torch.from_numpy(x_re)
        imag_data = torch.from_numpy(x_im)
        torch_data = torch.complex(real_data, imag_data)   
    
    window_data = torch.from_numpy(window["value"])

    golden = torch.torch.stft(torch_data, n_fft, hop_length=hop_length, win_length=win_length, window=window_data, center=center, pad_mode=pad_mode, onesided=onesided, normalized=normalized, return_complex=True)

    golden = golden.numpy()
    golden = np.array([np.real(golden), np.imag(golden)]).transpose(1, 2, 3, 0)
    golden = golden.reshape(y["shape"])

    return golden


def fuzz_branch_common(x, window, n_fft, hop_length, win_length, center, pad_mode = "reflect", normalized = False, onesided = False):
    
    x_shape = x["shape"]
    batch_num = x_shape[0]
    n = x_shape[1]

    dtype = x["type"]
    fuzz_value_x = (10*(np.random.randn(*x_shape).astype(dtype)))

    if onesided:
        fuzz_value_x[:,:,1] = np.zeros(x_shape[0]*x_shape[1]).astype(dtype).reshape(x_shape[0],x_shape[1])

    fuzz_value_x = fuzz_value_x.tolist()

    
    win_dtype = window["type"]
    window_data = np.random.randn(win_length).astype(win_dtype).tolist()

    
    if center:
        frame_count = n // hop_length + 1
    else:
        frame_count = (n - n_fft) // hop_length + 1
    
    
    if onesided:
        #x_shape = [n]
        y_shape = [batch_num, n_fft//2 + 1 , frame_count, 2]
    else:
        y_shape = [batch_num, n_fft, frame_count, 2]




    return {"input_desc": 
            {"x": 
             {
                "shape": x_shape,
                "value": fuzz_value_x
             },
             "window": 
             {
                "shape": [win_length],
                "value": window_data
             }
            },
            "output_desc":
            {"y": 
             {
                "shape": y_shape
             }
            },
            "attr":
            {"n_fft": 
             {
                "value": n_fft
             },
             "hop_length": 
             {
                "value": hop_length
             },
             "win_length": 
             {
                "value": win_length
             },
             "center": 
             {
                "value": center
             },
             "pad_mode": 
             {
                "value": pad_mode
             },
             "normalized": 
             {
                "value": normalized
             },
             "onesided": 
             {
                "value": onesided
             }
            }
            }


def fuzz_branch_stft_n2048_fft32_win32():
    x_shape = [1, 2048, 2]
    win_shape = [32,]
    n_fft = 64
    hop_length = 128
    win_length = win_shape[0]
    center = False
    pad_mode = "reflect"
    normalized = False
    onesided = False

    dtype = "float32"

    x = {"shape": x_shape,
             "type": dtype}
    
    window = {"shape": win_shape,
              "type": dtype}

    return fuzz_branch_common(x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)


def fuzz_branch_stft_n2048_fft1024_win32():
    x_shape = [1, 2048, 2]
    win_shape = [32,]
    n_fft = 1024
    hop_length = 128
    win_length = win_shape[0]
    center = False
    pad_mode = "reflect"
    normalized = False
    onesided = False

    dtype = "float32"

    x = {"shape": x_shape,
             "type": dtype}
    
    window = {"shape": win_shape,
              "type": dtype}

    return fuzz_branch_common(x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)


def fuzz_branch_stft_n2048_fft32_win32_centered_reflect():
    x_shape = [1, 2048, 2]
    win_shape = [32,]
    n_fft = 64
    hop_length = 128
    win_length = win_shape[0]
    center = True
    pad_mode = "reflect"
    normalized = False
    onesided = False

    dtype = "float32"

    x = {"shape": x_shape,
             "type": dtype}
    
    window = {"shape": win_shape,
              "type": dtype}

    return fuzz_branch_common(x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)


def fuzz_branch_stft_n2048_fft32_win32_centered_constant():
    x_shape = [1, 2048, 2]
    win_shape = [32,]
    n_fft = 64
    hop_length = 128
    win_length = win_shape[0]
    center = True
    pad_mode = "constant"
    normalized = False
    onesided = False

    dtype = "float32"

    x = {"shape": x_shape,
             "type": dtype}
    
    window = {"shape": win_shape,
              "type": dtype}

    return fuzz_branch_common(x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)


def fuzz_branch_stft_n2048_fft32_win32_centered_replicate():
    x_shape = [1, 2048, 2]
    win_shape = [32,]
    n_fft = 64
    hop_length = 128
    win_length = win_shape[0]
    center = True
    pad_mode = "replicate"
    normalized = False
    onesided = False

    dtype = "float32"

    x = {"shape": x_shape,
             "type": dtype}
    
    window = {"shape": win_shape,
              "type": dtype}

    return fuzz_branch_common(x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)


def fuzz_branch_stft_n2048_fft32_win32_centered_circular():
    x_shape = [1, 2048, 2]
    win_shape = [32,]
    n_fft = 64
    hop_length = 128
    win_length = win_shape[0]
    center = True
    pad_mode = "circular"
    normalized = False
    onesided = False

    dtype = "float32"

    x = {"shape": x_shape,
             "type": dtype}
    
    window = {"shape": win_shape,
              "type": dtype}

    return fuzz_branch_common(x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)


def fuzz_branch_stft_n2048_fft32_win32_normalized():
    x_shape = [1, 2048, 2]
    win_shape = [32,]
    n_fft = 64
    hop_length = 128
    win_length = win_shape[0]
    center = False
    pad_mode = "reflect"
    normalized = True
    onesided = False

    dtype = "float32"

    x = {"shape": x_shape,
             "type": dtype}
    
    window = {"shape": win_shape,
              "type": dtype}

    return fuzz_branch_common(x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)


def fuzz_branch_stft_n2048_fft32_win32_onesided():
    x_shape = [1, 2048, 2]
    win_shape = [32,]
    n_fft = 64
    hop_length = 128
    win_length = win_shape[0]
    center = False
    pad_mode = "reflect"
    normalized = False
    onesided = True

    dtype = "float32"

    x = {"shape": x_shape,
             "type": dtype}
    
    window = {"shape": win_shape,
              "type": dtype}

    return fuzz_branch_common(x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)


def fuzz_branch_stft_n2048_fft32_win32_centered_circular_normalized_onesided():
    x_shape = [1, 2048, 2]
    win_shape = [32,]
    n_fft = 64
    hop_length = 128
    win_length = win_shape[0]
    center = True
    pad_mode = "circular"
    normalized = True
    onesided = True

    dtype = "float32"

    x = {"shape": x_shape,
             "type": dtype}
    
    window = {"shape": win_shape,
              "type": dtype}

    return fuzz_branch_common(x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)


def fuzz_branch_stft_n2000_fft400_win400():
    x_shape = [2, 2000, 2]
    win_shape = [400,]
    n_fft = 400
    hop_length = 200
    win_length = win_shape[0]
    center = False
    pad_mode = "reflect"
    normalized = False
    onesided = False

    dtype = "float32"

    x = {"shape": x_shape,
             "type": dtype}
    
    window = {"shape": win_shape,
              "type": dtype}

    return fuzz_branch_common(x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)


def fuzz_branch_stft_n8192_fft8192_win8192():
    x_shape = [2, 8192, 2]
    win_shape = [8192,]
    n_fft = 8192
    hop_length = 8192
    win_length = win_shape[0]
    center = False
    pad_mode = "reflect"
    normalized = False
    onesided = False

    dtype = "float32"

    x = {"shape": x_shape,
             "type": dtype}
    
    window = {"shape": win_shape,
              "type": dtype}

    return fuzz_branch_common(x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)
