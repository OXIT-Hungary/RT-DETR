import torch
import torch.nn as nn

from .core import YAMLConfig


def get_model(args) -> nn.Module:
    """"""
    cfg = YAMLConfig(
        cfg_path="/home/geri/work/OXIT-Sport_Framework/src/submodules/rt-detr/rtdetrv2/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    return model
