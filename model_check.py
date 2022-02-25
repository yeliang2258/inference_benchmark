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

import argparse
import os
import logging
import yaml

from benchmark import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False


class ModelChecker():
    def __init__(self, args):
        self.args = args
        self.runner = BenchmarkRunner()
        print("test yaml file path: ", self.args.config_file)

    def onnx_config(self):
        self.args.return_result = True
        self.args.backend_type = "onnxruntime"

    def paddle_config(self):
        self.args.return_result = True
        self.args.backend_type = "paddle"

    def compare(self, result, expect, delta=1e-5, rtol=1e-5):
        if type(result) == np.ndarray:
            if type(expect) == list:
                expect = expect[0]
            expect = np.array(expect)
            res = np.allclose(
                result, expect, atol=delta, rtol=rtol, equal_nan=True)
            # 出错打印错误数据
            if res is False:
                diff = abs(result - expect)
                logging.error("Output has diff! max diff: {}".format(
                    np.amax(diff)))
            if result.dtype != expect.dtype:
                logging.error(
                    "Different output data types! res type is: {}, and expect type is: {}".
                    format(result.dtype, expect.dtype))
            failed_type = []
            if not res:
                failed_type.append(" results has diff ")
            if not result.shape == expect.shape:
                failed_type.append(" shape is not equal ")
            if not result.dtype == expect.dtype:
                failed_type.append(" dtype is not equal ")
            return failed_type
        elif isinstance(result, (list, tuple)):
            for i in range(len(result)):
                if isinstance(result[i], (np.generic, np.ndarray)):
                    return self.compare(result[i], expect[i], delta, rtol)
                else:
                    return self.compare(result[i].numpy(), expect[i], delta,
                                        rtol)

    def run(self):
        convert_sucess = os.path.exists(self.args.model_dir + "/model.onnx")
        if not convert_sucess:
            with open("result.txt", 'a+') as f:
                f.write(self.args.model_dir + ": convert failed! \n")
            return
        print(">>>> start check model diff ...... ")

        self.paddle_config()
        expect_result = self.runner.test(self.args)

        self.onnx_config()
        onnx_pred = self.runner.test(self.args)
        failed_type = self.compare(onnx_pred, expect_result)
        with open("result.txt", 'a+') as f:
            if not len(failed_type):
                f.write(self.args.model_dir + ": convert success! \n")
            for i in range(len(failed_type)):
                f.write(self.args.model_dir + ": " + failed_type[i] + "\n")
        print(">>>> end check model diff ! ")


def main():
    args = parse_args()

    checker = ModelChecker(args)
    checker.run()


if __name__ == "__main__":
    main()
