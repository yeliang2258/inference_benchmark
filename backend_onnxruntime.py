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

import backend
import os
import sys
import numpy as np
from common import getdtype

try:
    import onnxruntime as ort
except Exception as e:
    sys.stderr.write("Cannot import onnxruntime, maybe it's not installed.\n")


class BackendOnnxruntime(backend.Backend):
    def __init__(self):
        super(BackendOnnxruntime, self).__init__()
        self.h2d_time = []
        self.compute_time = []
        self.d2h_time = []

    def version(self):
        # paddle.version.commit
        return ort.__version__

    def name(self):
        return "onnxruntime"

    def load(self, config_arg, inputs=None, outpus=None):
        self.args = config_arg
        if os.path.exists(self.args.model_dir):
            model_file = os.path.join(self.args.model_dir, "model.onnx")
        else:
            raise ValueError(
                f"The model dir {self.args.model_dir} does not exists!")

        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = self.args.cpu_threads
        if self.args.enable_profile:
            sess_opt.enable_profiling = True

        if self.args.enable_openvino and not self.args.enable_gpu:
            run_providers = ['OpenVINOExecutionProvider']

        if not self.args.enable_mkldnn and not self.args.enable_gpu:
            run_providers = ['CPUExecutionProvider']

        if self.args.enable_mkldnn and not self.args.enable_gpu:
            run_providers = ['DnnlExecutionProvider']

        if self.args.enable_gpu:
            trt_providers = [('TensorrtExecutionProvider', {
                'device_id': self.args.gpu_id,
                'trt_max_workspace_size': 1073741824,
                'trt_fp16_enable': True
                if self.args.precision == 'fp16' else False,
            }), ('CUDAExecutionProvider', {
                'device_id': self.args.gpu_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            })]

            cuda_providers = [
                ('CUDAExecutionProvider', {
                    'device_id': self.args.gpu_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider',
            ]

            if self.args.enable_trt:
                run_providers = trt_providers
            else:
                run_providers = cuda_providers

        self.sess = ort.InferenceSession(
            model_file, sess_options=sess_opt, providers=run_providers)

        return self

    def reset(self):
        self.h2d_time.clear()
        self.d2h_time.clear()
        self.compute_time.clear()

    def warmup(self):
        pass

    def predict(self, feed=None):
        # prepare input
        input_data = {}
        for i in range(len(self.sess.get_inputs())):
            name = self.sess.get_inputs()[i].name
            if self.args.yaml_config["input_shape"][str(i)]["shape"][
                    self.args.test_num][0] == -1:
                input_shape = [self.args.batch_size] + self.args.yaml_config[
                    "input_shape"][str(i)]["shape"][self.args.test_num][1:]
                dtype = self.args.yaml_config["input_shape"][str(i)]["dtype"][
                    self.args.test_num]
            else:
                input_shape = self.args.yaml_config["input_shape"][str(i)][
                    "shape"][self.args.test_num]
                dtype = self.args.yaml_config["input_shape"][str(i)]["dtype"][
                    self.args.test_num]
            if hasattr(self.args, "test_data"):
                fake_input = self.args.test_data[i].astype(getdtype(dtype))
            else:
                fake_input = np.ones(input_shape, dtype=getdtype(dtype))
            input_data[name] = fake_input
        output = self.sess.run(None, input_data)
        if self.args.return_result:
            return output


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
    runner.report()
