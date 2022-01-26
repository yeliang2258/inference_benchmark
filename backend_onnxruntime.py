import backend
import os
import sys
import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    sys.stderr.write("Cannot import onnxruntime, maybe it's not installed.\n")


class BackendOnnxruntime(backend.Backend):

    def __init__(self):
        super(BackendOnnxruntime, self).__init__()

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

        if self.args.enable_gpu:
            self.sess = ort.InferenceSession(
                model_file, providers=['CUDAExecutionProvider'])

        return self

    def warmup(self):
        pass

    def predict(self, feed=None):
        # self.sess.run(output_names, {input_name: input_data})
        pass


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
    runner.report()
