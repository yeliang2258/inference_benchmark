#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import os
import time
import subprocess
import signal
import sys

import numpy as np
import yaml
import argparse
import psutil

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


class GPUStatBase(object):
    nvidia_smi_path = "nvidia-smi"
    keys = ('index', 'uuid', 'name', 'timestamp', 'memory.total',
            'memory.free', 'memory.used', 'utilization.gpu',
            'utilization.memory')
    nu_opt = ',nounits'


class GPUStat(GPUStatBase):
    def __init__(self, gpu_id=0):
        self.result = {}
        self.gpu_id = gpu_id

    def start(self):
        cmd = '%s --id=%s --query-gpu=%s --format=csv,noheader%s -lms 100' % (
            GPUStatBase.nvidia_smi_path, self.gpu_id,
            ','.join(GPUStatBase.keys), GPUStatBase.nu_opt)
        # print(cmd)
        self.routine = subprocess.Popen(
            cmd,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            shell=True,
            close_fds=True,
            preexec_fn=os.setsid)
        time.sleep(1.0)

    def stop(self):
        try:
            # os.killpg(p.pid, signal.SIGTERM)
            os.killpg(self.routine.pid, signal.SIGUSR1)
        except Exception as e:
            print(e)
            return

        lines = self.routine.stdout.readlines()
        # print(lines)
        lines = [
            line.strip().decode("utf-8") for line in lines
            if line.strip() != ''
        ]
        gpu_info_list = [{
            k: v
            for k, v in zip(GPUStatBase.keys, line.split(', '))
        } for line in lines]
        result = gpu_info_list[0]
        for item in gpu_info_list:
            for k in item.keys():
                result[k] = max(result[k], item[k])
        self.result = result

    def output(self):
        return self.result


def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False


def str2list(v):
    if len(v) == 0:
        return []

    return [list(map(int, item.split(","))) for item in v.split(":")]


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--type', required=True, choices=["cls", "shitu"])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--input_shape', type=str2list, default=[])
    parser.add_argument('--cpu_threads', type=int, default=1)
    parser.add_argument('--precision', type=str, choices=["fp32", "fp16"])
    parser.add_argument(
        '--backend_type',
        type=str,
        choices=["paddle", "onnxruntime"],
        default="paddle")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument(
        '--paddle_model_file', type=str, default="model.pdmodel")
    parser.add_argument(
        '--paddle_params_file', type=str, default="model.pdiparams")
    parser.add_argument('--enable_mkldnn', type=str2bool, default=True)
    parser.add_argument('--enable_openvino', type=str2bool, default=False)
    parser.add_argument('--enable_gpu', type=str2bool, default=False)
    parser.add_argument('--enable_trt', type=str2bool, default=False)
    parser.add_argument('--enable_profile', type=str2bool, default=False)
    parser.add_argument('--enable_benchmark', type=str2bool, default=True)
    parser.add_argument('--return_result', type=str2bool, default=False)
    parser.add_argument(
        '--config_file', type=str, required=True, default="config/model.yaml")
    args = parser.parse_args()
    return args


def get_backend(backend):
    if backend == "paddle":
        from backend_paddle import BackendPaddle
        backend = BackendPaddle()
    elif backend == "onnxruntime":
        from backend_onnxruntime import BackendOnnxruntime
        backend = BackendOnnxruntime()
    else:
        raise ValueError("unknown backend: " + backend)
    return backend


def parse_time(time_data, result_dict):
    percentiles = [50., 80., 90., 95., 99., 99.9]
    buckets = np.percentile(time_data, percentiles).tolist()
    buckets_str = ",".join(
        ["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])
    # if result_dict["total"] == 0:
    result_dict["total"] = len(time_data)
    result_dict["result"] = {
        str(k): float(format(v, '.6f'))
        for k, v in zip(percentiles, buckets)
    }
    avg_cost = np.mean(time_data)
    result_dict["result"]['avg_cost'] = float(format(avg_cost, '.6f'))


def parse_config(conf):
    return config


class BenchmarkRunner():
    def __init__(self):
        self.warmup_times = 20
        self.run_times = 100
        self.time_data = []
        self.backend = None
        self.conf = None

    def preset(self):
        self.backend = get_backend(self.conf.backend_type)
        self.backend.load(self.conf)
        self.gpu_stat = GPUStat(self.conf.gpu_id)
        self.gpu_stat.start()

    def run(self):
        if self.conf.return_result:
            output = self.backend.predict()
            return output

        for i in range(self.warmup_times):
            self.backend.predict()

        for i in range(self.run_times):
            begin = time.time()
            self.backend.predict()
            self.time_data.append(time.time() - begin)

    def report(self):
        self.gpu_stat.stop()
        perf_result = {}
        parse_time(self.time_data, perf_result)

        print('##### benchmark result: #####')
        result = {}
        result['detail'] = perf_result
        result['gpu_stat'] = self.gpu_stat.output()
        result['backend_type'] = self.conf.backend_type
        result['batch_size'] = self.conf.batch_size
        result['precision'] = self.conf.precision
        result['enable_gpu'] = self.conf.enable_gpu
        result['enable_trt'] = self.conf.enable_trt
        print(result)
        with open("result.txt", 'a+') as f:
            f.write("model path: " + self.conf.model_dir + "\n")
            for key, val in result.items():
                f.write(key + " : " + str(val) + "\n")
            f.write("\n")

    def test(self, conf):
        self.conf = conf
        config_path = os.path.abspath(self.conf.model_dir + "/" +
                                      self.conf.config_file)
        if not os.path.exists(config_path):
            log.error("{} not found".format(config_path))
            sys.exit(1)
        try:
            fd = open(config_path)
        except Exception as e:
            raise ValueError("open config file failed.")
        yaml_config = yaml.load(fd, yaml.FullLoader)
        fd.close()
        self.conf.yaml_config = yaml_config

        if self.conf.return_result:
            self.conf.test_num = 0
            self.preset()
            return self.run()

        test_num = len(self.conf.yaml_config["input_shape"]["0"]["shape"])
        for i in range(test_num):
            self.conf.test_num = i
            self.preset()
            self.run()
            self.report()


def main():
    args = parse_args()
    runner = BenchmarkRunner()
    runner.test(args)


if __name__ == "__main__":
    main()
