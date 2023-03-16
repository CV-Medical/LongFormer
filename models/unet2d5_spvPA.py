# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn

from models.params.networks.blocks.convolutions import Convolution, ResidualUnit
from models.params.networks.blocks.attentionblock import AttentionBlock1, AttentionBlock2

from monai.networks.layers.factories import Norm, Act

class UNet2d5_spvPA(nn.Module):
    def __init__(
        self,
        dimensions,
        in_channels,
        out_channels,
        channels,
        strides,
        kernel_sizes,
        sample_kernel_sizes,
        num_res_units=0,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0,
        attention_module=True,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == (len(strides)) + 1 == len(sample_kernel_sizes) + 1
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.sample_kernel_sizes = sample_kernel_sizes
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.attention_module = attention_module
        self.att_maps = []
        self.fea_maps = []
        
        def _create_block(inc, outc, channels, strides, kernel_sizes, sample_kernel_sizes, is_top):
            c = channels[0]
            s = strides[0]
            k = kernel_sizes[0]
            sk = sample_kernel_sizes[0]

            down = self._get_down_layer(in_channels=inc, out_channels=c, kernel_size=k)

            downsample = self._get_downsample_layer(in_channels=c, out_channels=c, strides=s, kernel_size=sk)

            if len(channels) > 2:
                subblock = _create_block(
                    c, channels[1], channels[1:], strides[1:], kernel_sizes[1:], sample_kernel_sizes[1:], is_top=False
                )
            else:
                subblock = self._get_bottom_layer(
                    in_channels=c,
                    out_channels=channels[1],
                    kernel_size=kernel_sizes[1],
                )

            subblock_with_resampling = nn.Sequential(downsample, subblock)
            return nn.Sequential(down, subblock_with_resampling)

        self.model = _create_block(
            in_channels, out_channels, self.channels, self.strides, self.kernel_sizes, self.sample_kernel_sizes, True
        )

        # register forward hooks on all Attentionblock1 modules, to save the attention maps
        if self.attention_module:
            for layer in self.model.modules():
                if type(layer) == AttentionBlock1:
                    layer.register_forward_hook(self.hook_save_attention_map)

    def hook_save_attention_map(self, module, inp, outp):
        if len(self.att_maps) == len(self.channels):
            self.att_maps = []
            self.fea_maps = []
        self.att_maps.append(outp[0])  # get first element of output (Attentionblock1 returns (att, x) )
        self.fea_maps.append(outp[1])


    def _get_att_layer(self, in_channels, out_channels, kernel_size):
        att1 = AttentionBlock1(self.dimensions, in_channels, out_channels, kernel_size, norm=None, dropout=self.dropout)
        att2 = AttentionBlock2(self.dimensions, in_channels, out_channels, kernel_size, norm=None, dropout=self.dropout)

        return nn.Sequential(att1, att2)

    def _get_down_layer(self, in_channels, out_channels, kernel_size):

        ru = ResidualUnit(
            self.dimensions,
            in_channels,
            out_channels,
            strides=1,
            kernel_size=kernel_size,
            subunits=self.num_res_units,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            )
        if self.attention_module:
            att_layer = self._get_att_layer(out_channels, out_channels, kernel_size)
            return nn.Sequential(ru,att_layer)
        else:
            return ru

    def _get_downsample_layer(self, in_channels, out_channels, strides, kernel_size):
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            kernel_size,
            self.act,
            self.norm,
            self.dropout,
            is_transposed=False,
        )
        return conv

    def _get_bottom_layer(self, in_channels, out_channels, kernel_size):
        conv = self._get_down_layer(in_channels, out_channels, kernel_size)
        return conv

    def forward(self, x):
        x = x.flatten(0,1) # b t 1 h w d -> b*t 1 h w d
        x1 = self.model(x)
        return x1, self.fea_maps[2:6]
