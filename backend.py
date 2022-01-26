class Backend():

    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.config = None

    def version(self):
        raise NotImplementdError("Backend:version")

    def name(self):
        raise NotImplementdError("Backend:name")

    def load(self, config_arg, inputs=None, outputs=None):
        raise NotImplementdError("Backend:load")

    def predict(self, feed):
        raise NotImplementdError("Backend:predict")

    def warmup(self):
        raise NotImplementdError("Backend:warmup")

    def get_performance_metrics(self):
        pass
