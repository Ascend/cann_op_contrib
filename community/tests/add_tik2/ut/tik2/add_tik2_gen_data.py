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

import numpy as np
import os


def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')


np.random.seed(677)

one_repeat_calcount = 128
block_dim_imm = 8
tile_num_imm = 8
double_buffer_imm = 2
total_length_imm = block_dim_imm * \
    one_repeat_calcount * tile_num_imm * double_buffer_imm

block_dim = np.array(block_dim_imm, dtype=np.uint32)
total_length = np.array(total_length_imm, dtype=np.uint32)
tile_num = np.array(tile_num_imm, dtype=np.uint32)
tiling = (block_dim, total_length, tile_num)
tiling_data = b''.join(x.tobytes() for x in tiling)

input_x = np.random.uniform(-100, 100, [total_length_imm,]).astype(np.float16)
input_y = np.random.uniform(-100, 100, [total_length_imm,]).astype(np.float16)
golden = (input_x + input_y).astype(np.float16)

write_file_txt("add_tik2/data/golden.txt", golden, fmt="%s")
with open('add_tik2/data/tiling.bin', "wb") as f:
    f.write(tiling_data)

input_x.tofile("add_tik2/data/input_x.bin")
write_file_txt("add_tik2/data/input_x.txt", input_x, fmt="%s")
input_y.tofile("add_tik2/data/input_y.bin")
write_file_txt("add_tik2/data/input_y.txt", input_y, fmt="%s")
