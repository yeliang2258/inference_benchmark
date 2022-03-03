# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import argparse
import pandas as pd

def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="result.txt",
                        help="tipc benchmark log path")
    parser.add_argument("--output_name", type=str, default="tipc_benchmark_excel.xlsx",
                        help="output excel file name")
    return parser.parse_args()


def log_split(file_name: str) -> list:
    """
    log split
    """
    log_list = []
    with open(file_name, 'r') as f:
        log_lines = f.read().split("model path")
        for log_line in log_lines:
            log_list.append(log_line)

    return log_list


def process_log(log_list: list) -> dict:
    """
    process log to dict
    """
    output_dict = {}

    for log_line in log_list.split("\n"):

        if "model_name" in log_line:
            output_dict["model_name"] = log_line.split(" : ")[-1].strip()
            continue
        if "avg_cost" in log_line:
            output_dict["avg_cost"] = log_line.split(" : ")[-1].strip()
            continue
        if "device_name" in log_line:
            output_dict["device_name"] = log_line.split(" : ")[-1].strip()
            continue
        if "gpu_mem" in log_line:
            output_dict["gpu_mem"] = log_line.split(" : ")[-1].strip()
            continue
        if "backend_type" in log_line:
            output_dict["backend_type"] = log_line.split(" : ")[-1].strip()
            continue
        if "batch_size" in log_line:
            output_dict["batch_size"] = log_line.split(" : ")[-1].strip()
            continue
        if "precision" in log_line:
            output_dict["precision"] = log_line.split(" : ")[-1].strip()
            continue
        if "enable_gpu" in log_line:
            output_dict["enable_gpu"] = log_line.split(" : ")[-1].strip()
            continue
        if "enable_trt" in log_line:
            output_dict["enable_trt"] = log_line.split(" : ")[-1].strip()
            continue
        else:
            continue

    return output_dict


def main(result_path, tipc_benchmark_excel_path):
    """
    main
    """
    # create empty DataFrame
    origin_df = pd.DataFrame(columns=["model_name", "avg_cost",
                                      "device_name", "gpu_mem",
                                      "backend_type","batch_size",
                                      "precision","enable_gpu",
                                      "enable_trt"])

    log_list = log_split(result_path)
    for one_model_log in log_list:
        output_total_list = process_log(one_model_log)
        origin_df = origin_df.append(output_total_list, ignore_index=True)

    raw_df = origin_df.sort_values(by='model_name')
    raw_df.to_excel(tipc_benchmark_excel_path)


if __name__ == "__main__":
    args = parse_args()
    result_path = args.result_path
    tipc_benchmark_excel_path = args.output_name
    main(result_path, tipc_benchmark_excel_path)
