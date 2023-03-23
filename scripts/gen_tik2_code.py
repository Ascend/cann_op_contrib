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
import os
from op_gen.template.tik2.scripts import tik2_impl_build
from op_gen.template.tik2.scripts import tik2_replay_build


def gen_op_code(ini_path, impl_path):
    try:
        soc_version = os.environ["AICPU_SOC_VERSION"]
    except KeyError:
        soc_version = "ascend910"
    file_out_path = "./build/install/community/op_impl"
    if not os.path.isdir(file_out_path):
        os.makedirs(file_out_path)
    tik2_replay_build.gen_replay(ini_path, impl_path, file_out_path, soc_version)
    tik2_impl_build.write_scripts(ini_path, file_out_path, soc_version)


def get_file_list():
    aicore_files = []
    for dirpath, dirnames, filenames in os.walk("community/ops"):
        for file_cpp in filenames:
            file_path = os.path.join(dirpath, file_cpp)
            if "ai_core" in file_path:
                aicore_files.append(file_path)
    return aicore_files


def get_op_name(file_name):
    list_path_split = file_name.split("/")
    return list_path_split[2]


def get_ini_path(file_list, key_str):
    for file_ini_path in file_list:
        if key_str in file_ini_path and ".ini" in file_ini_path:
            return file_ini_path
    return None


def gen_op_code_by_list(file_list):
    for file_cpp_path in file_list:
        if ".cpp" in file_cpp_path:
            op_name = get_op_name(file_cpp_path)
            ini_path = get_ini_path(file_list, op_name)
            impl_path = os.path.dirname(file_cpp_path)
            gen_op_code(ini_path, impl_path)


if __name__ == '__main__':
    gen_op_code_by_list(get_file_list())
