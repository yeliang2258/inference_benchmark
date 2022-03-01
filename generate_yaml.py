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

import yaml
import json
import sys
import ast
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="input.conf")
    parser.add_argument('--yaml_file', type=str, default="config.yaml")
    args = parser.parse_args()
    return args


def generate_yaml(data, config_file):
    config = {}
    shape_info = {}
    data = data.split(';')
    shape_dict = {}
    for item in data:
        shape_list = item.strip('][').strip('{}').split('},{')
        for i, shape in enumerate(shape_list):
            if str(i) not in shape_dict:
                shape_dict[str(i)] = {}
                shape_dict[str(i)]['dtype'] = []
                shape_dict[str(i)]['shape'] = []
            arr = shape.strip('][').split(',[')
            dtype, shape = arr[0], list(map(int, arr[1].split(',')))
            shape_dict[str(i)]['dtype'].append(dtype)
            shape_dict[str(i)]['shape'].append(shape)

    config['input_shape'] = shape_dict

    with open(config_file, 'w') as fd:
        yaml.dump(config, fd, default_flow_style=None)
    print(" Generate yaml file: ", config_file, "\n")


def main():
    args = parse_args()

    try:
        fd = open(args.input_file)
    except Exception as e:
        raise ValueError("open config file failed.")

    lines = fd.readlines()
    for line in lines:
        if 'random_infer_input' in line:
            line = line.strip('random_infer_input:')
            generate_yaml(line.strip(), args.yaml_file)
            return
    print(" Generate yaml file failed. \n")


if __name__ == "__main__":
    main()
