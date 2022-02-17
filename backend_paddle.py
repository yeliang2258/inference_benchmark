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

try:
    import paddle
    import paddle.inference as paddle_infer
except Exception as e:
    sys.stderr.write("Cannot import paddle, maybe paddle is not installed.\n")


class BackendPaddle(backend.Backend):
    def __init__(self):
        super(BackendPaddle, self).__init__()

    def version(self):
        # paddle.version.commit
        return paddle.version.full_version

    def name(self):
        return "paddle"

    def load(self, config_arg, inputs=None, outpus=None):
        self.args = config_arg
        if os.path.exists(self.args.model_dir):
            # model_file = os.path.join(self.args.model_dir, "model.pdmodel")
            # model_params = os.path.join(self.args.model_dir, "model.pdiparams")
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
        config.switch_ir_debug()
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
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=paddle_infer.PrecisionType.Float32,
                    max_batch_size=20,
                    min_subgraph_size=3)
        #config.disable_glog_info()
        pass_builder = config.pass_builder()
        #pass_builder.append_pass('interpolate_mkldnn_pass')

        self.predictor = paddle_infer.create_predictor(config)

        input_shape = self.args.input_shape
        print(input_shape)
        if len(input_shape) <= 0:
            raise Exception("input shape is empty.")
        # _ins_shape = [self.args.batch_size] + list(map(int, input_shape.split(',')))
        # fake_input = np.ones(_ins_shape, dtype=np.float32)

        # set input tensor
        input_names = self.predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = self.predictor.get_input_handle(name)
            input_shape = [self.args.batch_size] + self.args.input_shape[i]
            fake_input = np.ones(input_shape, dtype=np.float32)
            input_tensor.reshape(input_shape)
            input_tensor.copy_from_cpu(fake_input.copy())

        return self

    def set_input(self):
        # set input tensor
        input_names = self.predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = self.predictor.get_input_handle(name)
            input_shape = [self.args.batch_size] + self.args.input_shape[i]
            fake_input = np.ones(input_shape, dtype=np.float32)
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

    def warmup(self):
        # for i range(self.args.warmup):
        #     self.predictor.run()
        pass

    def predict(self, feed=None):
        self.set_input()
        self.predictor.run()
        output = self.set_output()
        if self.args.return_result:
            return output


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
    runner.report()
