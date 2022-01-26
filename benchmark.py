
import logging

import os
import time
import numpy as np
import yaml
import argparse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

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
    
    def run(self):
        for i in range(self.warmup_times):
            self.backend.predict()
        
        for i in range(self.run_times):
            begin = time.time()
            self.backend.predict()
            self.time_data.append(time.time() - begin)

    def report(self):
        result = {}
        parse_time(self.time_data, result)
        print(result)
        pass

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
