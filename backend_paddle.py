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
import time
import numpy as np
from common import getdtype

try:
    import paddle
    import paddle.inference as paddle_infer
except Exception as e:
    sys.stderr.write("Cannot import paddle, maybe paddle is not installed.\n")


class BackendPaddle(backend.Backend):
    def __init__(self):
        super(BackendPaddle, self).__init__()
        self.h2d_time = []
        self.compute_time = []
        self.d2h_time = []

    def version(self):
        # paddle.version.commit
        return paddle.version.full_version

    def name(self):
        return "paddle"

    def load(self, config_arg, inputs=None, outpus=None):
        self.args = config_arg
        if os.path.exists(self.args.model_dir):
            model_file = os.path.join(self.args.model_dir + "/" +
                                      self.args.paddle_model_file)
            model_params = os.path.join(self.args.model_dir + "/" +
                                        self.args.paddle_params_file)
            config = paddle_infer.Config(model_file, model_params)
        else:
            raise ValueError(
                f"The model dir {self.args.model_dir} does not exists!")

        # enable memory optim
        config.enable_memory_optim()
        #config.disable_gpu()
        config.set_cpu_math_library_num_threads(self.args.cpu_threads)
        config.switch_ir_optim(True)
        # debug
        # config.switch_ir_debug()
        if self.args.enable_mkldnn and not self.args.enable_gpu:
            config.disable_gpu()
            config.enable_mkldnn()
        if not self.args.enable_mkldnn and not self.args.enable_gpu:
            config.disable_gpu()
            # config.enable_mkldnn()
        if self.args.enable_profile:
            config.enable_profile()
        if self.args.enable_gpu:
            config.enable_use_gpu(256, self.args.gpu_id)
            if self.args.enable_trt:
                max_batch_size = self.args.batch_size
                if self.args.yaml_config["input_shape"]["0"]["shape"][
                        self.args.test_num][0] != -1:
                    max_batch_size = self.args.yaml_config["input_shape"]["0"][
                        "shape"][self.args.test_num][0]
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=paddle_infer.PrecisionType.Float32,
                    max_batch_size=max_batch_size,
                    min_subgraph_size=3)
        #config.disable_glog_info()
        pass_builder = config.pass_builder()
        #pass_builder.append_pass('interpolate_mkldnn_pass')

        self.predictor = paddle_infer.create_predictor(config)

        input_shape = self.args.yaml_config["input_shape"]
        if len(input_shape) <= 0:
            raise Exception("input shape is empty.")

        return self

    def set_input(self):
        # set input tensor
        input_names = self.predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = self.predictor.get_input_handle(name)
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
            input_tensor.reshape(input_shape)
            input_tensor.copy_from_cpu(fake_input.copy())

    def set_output(self):
        results = []
        # get out data from output tensor
        output_names = self.predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = self.predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        if self.args.return_result:
            return results

    def reset(self):
        self.h2d_time.clear()
        self.d2h_time.clear()
        self.compute_time.clear()

    def warmup(self):
        # for i range(self.args.warmup):
        #     self.predictor.run()
        pass

    def predict(self, feed=None):
        h2d_begin = time.time()
        self.set_input()
        h2d_end = time.time()
        self.predictor.run()
        d2h_begin = time.time()
        output = self.set_output()
        d2h_end = time.time()
        self.h2d_time.append(h2d_end - h2d_begin)
        self.compute_time.append(d2h_begin - h2d_end)
        self.d2h_time.append(d2h_end - d2h_begin)
        if self.args.return_result:
            return output


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
    runner.report()
