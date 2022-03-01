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

import paddle
import paddle.nn as nn


class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._conv2d = nn.Conv2D(2, 2, 1)

    def forward(self, x, y):
        x = self._conv2d(x)
        return x + y


layer = LinearNet()
input_x = paddle.static.InputSpec(shape=[-1, 2, 3, 3])
input_y = paddle.static.InputSpec(shape=[-1, 2, 3, 3])
path = "./model"
paddle.jit.save(layer, path, input_spec=[input_x, input_y])