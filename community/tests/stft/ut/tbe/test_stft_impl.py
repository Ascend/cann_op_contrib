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

import sys
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info
import torch
import numpy as np
ut_case = BroadcastOpUT("stft")

def calc_expect_func(input, window, y, n_fft, hop_length=128, win_length=0, center=True, pad_mode="reflect", normalized=False, onesided=True,
                     return_complex=True):
    input_value = input["value"]
    if onesided:
        x_re = np.array(input_value[:,:,0])
        torch_data = torch.from_numpy(x_re)
    else:
        x_re = np.array(input_value[:,:,0])
        x_im = np.array(input_value[:,:,1])
        real_data = torch.from_numpy(x_re)
        imag_data = torch.from_numpy(x_im)
        torch_data = torch.complex(real_data, imag_data)
    window_data = torch.from_numpy(window["value"])
    golden = torch.stft(torch_data,n_fft,hop_length=hop_length,win_length=win_length,window=window_data,center=center,pad_mode=pad_mode,onesided=onesided,normalized=normalized,return_complex=return_complex)
    golden = golden.numpy()
    golden = np.array([np.real(golden),np.imag(golden)]).transpose(1,2,3,0)
    golden = golden.reshape(y["shape"])
    return [golden, ]


def stft_ut_py_test(x_shape,n_fft,hop_length,win_length,center,pad_mode = "reflect",normalized=False,onesided=False):
    batch_num = x_shape[0]
    n = x_shape[1]

    if center:
        frame_count = n // hop_length + 1
    else:
        frame_count = (n - n_fft) // hop_length + 1
    
    if onesided:
        y_shape = [batch_num, n_fft//2 + 1, frame_count, 2]
    else:
        y_shape = [batch_num, n_fft, frame_count, 2]


    ut_case.add_precision_case("all", {
        "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": x_shape, "shape": x_shape,
                    "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (win_length,), "shape": (win_length,),
                    "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": y_shape, "shape": y_shape,
                    "param_type": "output"},
                    n_fft,
                    hop_length,
                    win_length,
                    center,
                    pad_mode,
                    normalized,
                    onesided
                    ],
        "precision_standard": precision_info.PrecisionStandard(1, 1),
        "calc_expect_func": calc_expect_func
    })


x_shape = (1, 2048, 2)
n_fft = 64
hop_length = 64
win_length = 64
center = False
pad_mode = "reflect"
normalized = False
onesided = False
stft_ut_py_test(x_shape, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)

x_shape = (1, 512, 2)
n_fft = 16
hop_length = 16
win_length = 16
center = True
pad_mode = "reflect"
normalized = False
onesided = False
stft_ut_py_test(x_shape, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)

x_shape = (1, 512, 2)
n_fft = 16
hop_length = 16
win_length = 16
center = False
pad_mode = "reflect"
normalized = True
onesided = True
stft_ut_py_test(x_shape, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)

x_shape = (1, 512, 2)
n_fft = 16
hop_length = 16
win_length = 16
center = True
pad_mode = "circular"
normalized = False
onesided = False
stft_ut_py_test(x_shape, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)

x_shape = (1, 512, 2)
n_fft = 16
hop_length = 16
win_length = 16
center = True
pad_mode = "constant"
normalized = False
onesided = False
stft_ut_py_test(x_shape, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)

x_shape = (1, 512, 2)
n_fft = 16
hop_length = 16
win_length = 16
center = True
pad_mode = "replicate"
normalized = False
onesided = False
stft_ut_py_test(x_shape, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)

x_shape = (1, 512, 2)
n_fft = 16
hop_length = 16
win_length = 16
center = False
pad_mode = "reflect"
normalized = False
onesided = False
stft_ut_py_test(x_shape, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)

x_shape = (1, 512, 2)
n_fft = 16
hop_length = 16
win_length = 16
center = False
pad_mode = "circular"
normalized = False
onesided = False
stft_ut_py_test(x_shape, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)

x_shape = (1, 512, 2)
n_fft = 16
hop_length = 16
win_length = 16
center = False
pad_mode = "constant"
normalized = False
onesided = False
stft_ut_py_test(x_shape, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)

x_shape = (1, 512, 2)
n_fft = 16
hop_length = 16
win_length = 16
center = False
pad_mode = "replicate"
normalized = False
onesided = False
stft_ut_py_test(x_shape, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)

x_shape = (1, 512, 2)
n_fft = 400
hop_length = 160
win_length = 400
center = False
pad_mode = "replicate"
normalized = False
onesided = False
stft_ut_py_test(x_shape, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided)
