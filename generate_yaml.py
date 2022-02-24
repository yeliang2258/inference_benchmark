import yaml
import json
import sys
import ast
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file', type=str, default="input.conf")
    parser.add_argument(
        '--yaml_file', type=str, default="config.yaml")
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


def main():
    args = parse_args()

    try:
        fd = open(args.input_file)
    except Exception as e:
        raise ValueError("open config file failed.")

    lines = fd.readlines()
    for line in lines:
        print('infer_input:', line)
        if 'infer_input' in line:
            line = line.strip('infer_input:')
            generate_yaml(line.strip(), args.yaml_file)
            return

if __name__ == "__main__":
    main()
