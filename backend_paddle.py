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
            model_file = os.path.join(self.args.model_dir, "model.pdmodel")
            model_params = os.path.join(self.args.model_dir, "model.pdiparams")
            config = paddle_infer.Config(model_file, model_params)
        else:
            raise ValueError(
                f"The model dir {self.args.model_dir} does not exists!")

        # enable memory optim
        config.enable_memory_optim()
        #config.disable_gpu()
        config.set_cpu_math_library_num_threads(self.args.cpu_threads)
        config.switch_ir_optim(True)
        if self.args.enable_mkldnn and not self.args.enable_gpu:
            config.disable_gpu()
            config.enable_mkldnn()
        if self.args.enable_profile:
            config.enable_profile()
        if self.args.enable_gpu:
            config.enable_use_gpu(256, self.args.gpu_id)
            if self.args.enable_trt:
                config.enable_tensorrt_engine(
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

    def warmup(self):
        # for i range(self.args.warmup):
        #     self.predictor.run()
        pass

    def predict(self, feed=None):
        self.predictor.run()


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
    runner.report()
