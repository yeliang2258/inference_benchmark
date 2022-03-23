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
from common import getdtype, randtool

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
            diff = abs(result - expect)
            if res is False:
                logging.error("Output has diff! max diff: {}".format(
                    np.amax(diff)))
            if result.dtype != expect.dtype:
                logging.error(
                    "Different output data types! res type is: {}, and expect type is: {}".
                    format(result.dtype, expect.dtype))
            failed_type = []
            if not res:
                if np.isnan(diff).any():
                    failed_type.append(" results have Nan ")
                else:
                    failed_type.append(" results have diff ")
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

        config_path = os.path.abspath(self.args.model_dir + "/" +
                                      self.args.config_file)
        try:
            fd = open(config_path)
        except Exception as e:
            raise ValueError("open config file failed.")
        yaml_config = yaml.load(fd, yaml.FullLoader)
        fd.close()
        input_data = []
        for i, val in enumerate(yaml_config["input_shape"]):
            input_shape = yaml_config["input_shape"][val]
            shape = [1] + input_shape["shape"][0][1:]
            dtype = input_shape["dtype"][0]
            data = randtool(dtype, -1, 1, shape)
            input_data.append(data)
        self.args.test_data = input_data

        self.paddle_config()
        try:
            expect_result = self.runner.test(self.args)
        except Exception as e:
            with open("result.txt", 'a+') as f:
                f.write(self.args.model_dir + ": paddle infer failed! \n")
            raise ValueError(self.args.model_dir + ": paddle infer failed!")

        self.onnx_config()
        try:
            onnx_pred = self.runner.test(self.args)
        except Exception as e:
            with open("result.txt", 'a+') as f:
                f.write(self.args.model_dir + ": onnxruntime infer failed! \n")
            raise ValueError(self.args.model_dir +
                             ": onnxruntime infer failed!")

        failed_type = self.compare(onnx_pred, expect_result)
        with open("result.txt", 'a+') as f:
            if not len(failed_type):
                f.write(self.args.model_dir + ": convert success! \n")
                print(">>>> check model diff success! ")
                return
            for i in range(len(failed_type)):
                f.write(self.args.model_dir + ": " + failed_type[i] + "\n")
            print(">>>> check model diff failed! ")


def main():
    args = parse_args()

    checker = ModelChecker(args)
    checker.run()


if __name__ == "__main__":
    main()
