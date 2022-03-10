# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import time
import paddle
import argparse
import pandas as pd
import onnxruntime


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path",
        type=str,
        default="./result.txt",
        help="tipc benchmark log path")
    parser.add_argument(
        "--output_name",
        type=str,
        default="tipc_benchmark_excel.xlsx",
        help="output excel file name")
    parser.add_argument(
        "--docker_name",
        type=str,
        default="paddle_manylinux_devel:cuda11.2-cudnn8.2.1-trt8.0.3.4-gcc82",
        help="docker name")

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


def model2onnx_log(file_name: str) -> list:
    """
    model to onnx log split
    """
    model2onnx_log_list = []
    with open(file_name, 'r') as f:
        model2onnx_log = f.readlines()
        for log_line in model2onnx_log:
            if "convert" in log_line:
                model2onnx_log_list.append(log_line)
    return model2onnx_log_list


def process_log(log_list: list) -> dict:
    """
    process log to dict
    """
    output_dict = {}

    for log_line in log_list.split("\n"):

        if "model_name" in log_line:
            output_dict["model_name"] = log_line.split(" : ")[-1].strip()
            continue
        if "cpu_threads" in log_line:
            output_dict["cpu_threads"] = log_line.split(" : ")[-1].strip()
            continue
        if "enable_mkldnn" in log_line:
            output_dict["enable_mkldnn"] = log_line.split(" : ")[-1].strip()
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
        if "cpu_mem" in log_line:
            output_dict["cpu_mem"] = log_line.split(" : ")[-1].strip()
            continue
        else:
            continue
    if "device_name" not in output_dict.keys(
    ) and "model_name" in output_dict.keys():
        output_dict["device_name"] = "CPU"
        output_dict["gpu_mem"] = "0"
    return output_dict


def data_merging(env, paddle_version, onnxruntime_version,
                 model2onnx_name_list, output_total_list):
    log_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    data_merging_logs = []
    for output_dict_num in range(len(output_total_list)):
        output_dict = output_total_list[output_dict_num]
        log_dict = {}
        for comp_list in output_total_list[output_dict_num:]:
            if output_dict["model_name"] == comp_list["model_name"] and \
                    output_dict["device_name"] == comp_list["device_name"] and \
                    output_dict["batch_size"] == comp_list["batch_size"] and \
                    output_dict["precision"] == comp_list["precision"] and \
                    output_dict["cpu_threads"] == comp_list["cpu_threads"] and \
                    output_dict["enable_mkldnn"] == comp_list["enable_mkldnn"] and \
                    output_dict["enable_gpu"] == comp_list["enable_gpu"] and \
                    output_dict["enable_trt"] == comp_list["enable_trt"] and \
                    output_dict["backend_type"] != comp_list["backend_type"]:
                log_dict["日期"] = log_time
                log_dict["环境"] = env
                log_dict["paddle_version"] = paddle_version
                log_dict["onnxruntime_version"] = onnxruntime_version
                log_dict["device_name"] = output_dict["device_name"]
                log_dict["model_name"] = output_dict["model_name"]
                log_dict["precision"] = output_dict["precision"]
                log_dict["batch_size"] = output_dict["batch_size"]
                log_dict["enable_gpu"] = output_dict["enable_gpu"]
                log_dict["enable_mkldnn"] = output_dict["enable_mkldnn"]
                log_dict["cpu_threads"] = output_dict["cpu_threads"]
                log_dict["enable_trt"] = output_dict["enable_trt"]
                log_dict["onnxruntime_cpu_mem"] = comp_list["cpu_mem"]
                log_dict["paddle_cpu_mem"] = output_dict["cpu_mem"]
                log_dict["paddle_gpu_mem"] = output_dict["gpu_mem"]
                log_dict["onnxruntime_gpu_mem"] = comp_list["gpu_mem"]
                log_dict["paddle_avg_cost"] = output_dict["avg_cost"]
                log_dict["onnxruntime_avg_cost"] = comp_list["avg_cost"]
                log_dict["cpu_mem_gap"] = calculation_gap(
                    paddle_num=log_dict["paddle_cpu_mem"],
                    onnxruntime_num=log_dict["paddle_cpu_mem"])
                log_dict["gpu_mem_gap"] = calculation_gap(
                    paddle_num=log_dict["paddle_gpu_mem"],
                    onnxruntime_num=log_dict["onnxruntime_gpu_mem"])
                log_dict["perf_gap"] = calculation_gap(
                    paddle_num=log_dict["paddle_avg_cost"],
                    onnxruntime_num=log_dict["onnxruntime_avg_cost"])
                log_dict["paddle2onnx_model_convert"] = "Failed"
                for model2onnx_success_log in model2onnx_name_list:
                    if log_dict["model_name"] in model2onnx_success_log:
                        log_dict["paddle2onnx_model_convert"] = model2onnx_success_log.split("convert")[-1]
                        break
            else:
                continue
            data_merging_logs.append(log_dict)
    return data_merging_logs


def calculation_gap(paddle_num, onnxruntime_num):
    if float(paddle_num) <= 0 and float(onnxruntime_num) <= 0:
        return 0
    else:
        return str((float(paddle_num) - float(onnxruntime_num)) /
                   float(onnxruntime_num))


def main(args, result_path, tipc_benchmark_excel_path):
    """
    main
    """
    # create empty DataFrame
    env = args.docker_name
    paddle_commit = paddle.__git_commit__
    paddle_tag = paddle.__version__
    paddle_version = paddle_tag + "/" + paddle_commit
    onnxruntime_version = onnxruntime.__version__
    origin_df = pd.DataFrame(columns=[
        "日期", "环境", "paddle_version", "onnxruntime_version", "device_name",
        "model_name", "precision", "batch_size", "enable_mkldnn",
        "cpu_threads", "enable_gpu", "enable_trt", "paddle2onnx_model_convert",
        "onnxruntime_cpu_mem", "paddle_cpu_mem", "paddle_gpu_mem",
        "onnxruntime_gpu_mem", "onnxruntime_avg_cost", "paddle_avg_cost",
        "cpu_mem_gap", "gpu_mem_gap", "perf_gap"
    ])

    log_list = log_split(result_path)
    model2onnx_name_list = model2onnx_log(result_path)
    dict_list = []
    for one_model_log in log_list:
        output_total_list = process_log(one_model_log)
        if "model_name" in output_total_list.keys():
            dict_list.append(output_total_list)
    output_excl_list = data_merging(env, paddle_version, onnxruntime_version,
                                    model2onnx_name_list, dict_list)
    for one_log in output_excl_list:
        origin_df = origin_df.append(one_log, ignore_index=True)
    raw_df = origin_df.sort_values(by='device_name')
    raw_df.to_excel(tipc_benchmark_excel_path)


if __name__ == "__main__":
    args = parse_args()
    result_path = args.result_path
    tipc_benchmark_excel_path = args.output_name
    main(args, result_path, tipc_benchmark_excel_path)

