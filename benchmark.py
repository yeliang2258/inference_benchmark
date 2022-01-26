
import logging

import os
import time
import subprocess
import signal

import numpy as np
import yaml
import argparse
import psutil

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


class GPUStatBase(object):
    nvidia_smi_path="nvidia-smi"
    keys = (
                'index',
                'uuid',
                'name',
                'timestamp',
                'memory.total',
                'memory.free',
                'memory.used',
                'utilization.gpu',
                'utilization.memory'
            )
    nu_opt=',nounits'

class GPUStat(GPUStatBase):

    def __init__(self, gpu_id=0):
        self.result = {}
        self.gpu_id = gpu_id

    def start(self):
        cmd = '%s --id=%s --query-gpu=%s --format=csv,noheader%s -lms 100' % (GPUStatBase.nvidia_smi_path, self.gpu_id, ','.join(GPUStatBase.keys), GPUStatBase.nu_opt)
        # print(cmd)
        self.routine = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,shell=True,close_fds=True,preexec_fn=os.setsid)
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
        lines = [ line.strip().decode("utf-8") for line in lines if line.strip() != '' ]
        gpu_info_list = [ { k: v for k, v in zip(GPUStatBase.keys, line.split(', ')) } for line in lines ]
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
    
    return [ list(map(int, item.split(","))) for item in v.split(":")]

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--type', required=True, choices=["cls", "shitu"])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--input_shape', type=str2list, default=[])
    parser.add_argument('--cpu_threads', type=int, default=1)
    parser.add_argument('--precision', type=str, choices=["fp32", "fp16"])
    parser.add_argument('--backend_type', type=str, choices=["paddle", "onnxruntime"], default="paddle")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--enable_mkldnn', type=str2bool, default=True)
    parser.add_argument('--enable_gpu', type=str2bool, default=False)
    parser.add_argument('--enable_trt', type=str2bool, default=False)
    parser.add_argument('--enable_profile', type=str2bool, default=False)
    parser.add_argument('--enable_benchmark', type=str2bool, default=True)
    parser.add_argument(
        '--config_file',
        type=str,
        required=True,
        default="config/model.yaml")
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
    buckets_str = ",".join(["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])
    # if result_dict["total"] == 0:
    result_dict["total"] = len(time_data)
    result_dict["result"] = {str(k): float(format(v, '.4f')) for k, v in zip(percentiles, buckets)}

def parse_config(conf):
    return config

class BenchmarkRunner():
    def __init__(self):
        self.warmup_times = 20
        self.run_times = 100
        self.time_data = []
        self.backend = None

    def load(self, conf):
        self.backend = get_backend(conf.backend_type)
        self.backend.load(conf)
        self.gpu_stat = GPUStat(conf.gpu_id)
        self.gpu_stat.start() 
    
    def run(self):
        for i in range(self.warmup_times):
            self.backend.predict()
        
        for i in range(self.run_times):
            begin = time.time()
            self.backend.predict()
            self.time_data.append(time.time() - begin)

    def report(self):
        self.gpu_stat.stop() 
        result = {}
        parse_time(self.time_data, result)
        print("##### latency stat #####")
        print(result)
        print("##### GPU stat #####")
        print(self.gpu_stat.output())

def main():
    args = parse_args()
    config = os.path.abspath(args.config_file)
    if not os.path.exists(config):
        log.error("{} not found".format(config_file))
        sys.exit(1)

    try:
        fd = open(args.config_file)
    except Exception as e:
        raise ValueError("open config file failed.")
    
    config = yaml.load(fd, yaml.FullLoader)
    fd.close()

    runner = BenchmarkRunner()
    runner.load(args)
    runner.run()
    runner.report()

if __name__ == "__main__":
    main()
