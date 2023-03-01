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
import os
import sys
from op_test_frame.ut import op_ut_runner


def conf_path():
    op_path = "../community/ops/"
    op_list = os.listdir(op_path)
    op_impl_path_end = "ai_core/cust_impl"
    source_path = ""
    for file_path in op_list:
        abs_path = os.path.abspath(os.path.join(op_path, file_path))
        op_impl_path = os.path.join(abs_path, op_impl_path_end)
        if os.path.isdir(op_impl_path):
            source_path = source_path + ":" + op_impl_path
    os.environ["source_path"] = source_path
    os.environ["util_path"] = os.path.abspath("../community/common")
    os.environ["PYTHONPATH"] = os.environ["source_path"] + ":" + os.environ["PYTHONPATH"]


def run_case():
    case_files = "../community/tests"
    cov_report = "html"
    process_num = 1
    soc_version = 'Ascend910A'
    simulator_lib_path = os.environ["ASCEND_AICPU_PATH"] + '/toolkit/tools/simulator'
    cov_report_path = "./cov_report_tbe"
    simulator_mode = "pv"
    op_ut_run = os.environ["ASCEND_AICPU_PATH"] + "/toolkit/python/site-packages/bin/op_ut_run"
    cmd = "{} --case_files={} --cov_report={} --process_num={} --soc_version={} " \
          "--simulator_lib_path={} --simulator_mode={} --cov_report_path={}".format(
        op_ut_run, case_files, cov_report, process_num, soc_version, simulator_lib_path, simulator_mode,
        cov_report_path)
    res = os.system(cmd)

    if res == 0:
        report_path = "../build/cov_report_tbe/"
        if os.path.isdir(report_path):
            exit(res)
        else:
            exit(-1)
    else:
        exit(-1)


def clean_old_cov():
    report_path = "../build/cov_report_tbe/"
    if os.path.exists(report_path):
        cmd = "rm -rf {}".format(report_path)
        os.system(cmd)


if __name__ == "__main__":
    conf_path()
    clean_old_cov()
    run_case()