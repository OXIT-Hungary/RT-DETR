"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from pathlib import Path
import pickle
import cv2
import numpy as np
from PIL import Image

from src.core import YAMLConfig

color = [
    (255, 255, 0),
    (0, 0, 255),
    (25, 255, 0),
    (0, 255, 255),
]


def draw(frame, labels, boxes, scores):
    for i, bbox in enumerate(boxes):
        cv2.rectangle(frame, bbox[:2], bbox[2:4], color[labels[i] - 1], 2)

    img = cv2.resize(np.array(frame), (1280, 720))
    cv2.imshow("out", img)


def main(args):
    """main"""
    cfg = YAMLConfig(args.config, exp_name=args.exp_name, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # NOTE load train mode state -> convert to deploy mode
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
    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    video_cap = cv2.VideoCapture(args.im_file)

    bboxes = []
    while True:
        has_frame, frame = video_cap.read()
        if not has_frame:
            break
        # frame = frame[420:1150, 850:3660]
        h, w = frame.shape[:2]
        orig_size = torch.Tensor([w, h])[None].to(args.device)

        im_data = transforms(Image.fromarray(frame))[None].to(args.device)
        labels, boxes, scores = model(im_data, orig_size)

        boxes = boxes.type(torch.int16).detach().cpu().tolist()
        labels = labels.type(torch.int8).detach().cpu().tolist()
        bboxes.append(boxes)
        draw(frame, labels, boxes, scores)

        key = cv2.waitKey(10) & 0xFF
        if key == 27 or key == ord("q") or key == ord("Q"):
            break

    video_cap.release()

    with open(f"outputs/boxes/{Path(args.im_file).stem}.pkl", "wb") as f:
        pickle.dump(bboxes, f)


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-r", "--resume", type=str)
    parser.add_argument("-f", "--im-file", type=str)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-n", "--exp_name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    main(args)
