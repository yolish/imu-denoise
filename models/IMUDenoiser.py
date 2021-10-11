"""
General framework for imu-denoising
"""
from torch import nn
from models.backbones.CNNB import CNNB
from models.backbones.Transformer import Transformer


class IMUDenoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.denoising_type = config.get("denoising_type") # seq-to-seq, residual
        self.backbone_type = config.get("backbone_type")
        backbones_config = config.get("backbones")
        backbone_config = backbones_config.get(self.backbone_type)
        print(backbone_config)
        backbone_config["input_dim"] = config.get("input_dim")
        self.backbone = self.get_backbone(backbone_config)

    def get_backbone(self, backbone_config):
        if self.backbone_type == "cnnb":
            return CNNB(backbone_config)
        elif self.backbone_type == "transformer":
            return Transformer(backbone_config)
        else:
            raise NotImplementedError("Backbone {} not supported".format(self.backbone_type))

    def forward(self, data):
            orig_src = data.get('imu')  # Shape N x S x C with S = sequence length, N = batch size, C = channels
            if self.denoising_type == 'seq2seq':
                return self.backbone(orig_src)
            elif self.denoising_type == 'residual':
                return (orig_src + self.backbone(orig_src))
            else:
                raise NotImplementedError("Denoising type: {} not supported".format(self.denoising_type))


