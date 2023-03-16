from torch import nn

from .position_encoding import build_position_encoding
from .unet2d5_spvPA import UNet2d5_spvPA

from monai.networks.layers import Norm

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        
    def forward(self, tensor):
         
        out, features = self[0](tensor)
        pos = []
        for i in range(4):
             
            pos.append(self[1](features[i]))

        return features, pos


def build_backbone(args, num_classes):
    position_embedding = build_position_encoding(args)
    backbone = UNet2d5_spvPA(
            dimensions=3,
            in_channels=1,
            out_channels=num_classes,
            channels=(16, 32, 64, 80, 96, 128),
            strides=(
                (2, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
            ),
            kernel_sizes=(
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
            ),
            sample_kernel_sizes=(
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
            ),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.1,
            attention_module=True,
        )
    model = Joiner(backbone, position_embedding)
    return model

