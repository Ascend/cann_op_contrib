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
from op_gen.template.tik2.cmake.util import tik2_impl_build
from op_gen.template.tik2.cmake.util import tik2_replay_build
import const_var


def get_impl_list():
    aicore_files = []
    for dirpath, dirnames, filenames in os.walk("community/ops"):
        for file_cpp in filenames:
            file_path = os.path.join(dirpath, file_cpp)
            if "op_kernel" in file_path:
                aicore_files.append(os.path.dirname(file_path))
    return aicore_files


def gen_op_code(ini_path, impl_path):
    try:
        soc_version = os.environ["SOC_VERSION"]
    except KeyError:
        soc_version = "ascend910"

    try:
        custom_version = os.environ["OPP_CUSTOM_VENDOR"]
    except KeyError as e:
        custom_version = "community"
    file_out_path = "./build/ascendc/op_impl/ai_core/tbe/{}_impl".format(custom_version)
    if not os.path.isdir(file_out_path):
        os.makedirs(file_out_path)
    rep_cfg = {}
    rep_cfg[const_var.REPLAY_BATCH] = ""
    rep_cfg[const_var.REPLAY_ITERATE] = ""
    rep_dir = {}
    rep_dir[const_var.CFG_IMPL_DIR] = impl_path
    rep_dir[const_var.CFG_OUT_DIR] = file_out_path
    tik2_replay_build.gen_replay(ini_path, rep_cfg, rep_dir, soc_version)
    tik2_impl_build.write_scripts(ini_path, rep_cfg, rep_dir)

    for dirpath, dirnames, filenames in os.walk(file_out_path):
        for file_cpp in filenames:
            file_path = os.path.join(dirpath, file_cpp)
            if impl_path.split("/")[2] in file_path:
                copy_to_dir = os.path.dirname(file_path)
                cp_cmd = "cp {}/* {}".format(impl_path, copy_to_dir)
                os.system(cp_cmd)


def get_ini_list():
    ini_files = []
    ini_base_path = "./build/ascendc/"
    if not os.path.isdir(ini_base_path):
        os.makedirs(ini_base_path)
    all_file = os.listdir(ini_base_path)
    for file_name in all_file:
        if ".ini" in file_name:
            ini_files.append(os.path.abspath(ini_base_path + file_name))
    return ini_files


def gen_op_code_by_list(file_list, ini_list):
    for file_impl in file_list:
        for file_ini in ini_list:
            gen_op_code(file_ini, file_impl)


if __name__ == '__main__':
    gen_op_code_by_list(get_impl_list(), get_ini_list())
