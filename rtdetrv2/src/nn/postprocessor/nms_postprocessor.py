"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn.functional as F 
import torch.distributed
import torchvision
from torch import Tensor
from torchvision.ops import box_iou
from ...core import register

from typing import Dict 


__all__ = ['DetNMSPostProcessor', ]


@register()
class DetNMSPostProcessor(torch.nn.Module):
    def __init__(self, \
                iou_threshold=0.5,
                score_threshold=0.6,
                keep_topk=20,
                box_fmt='cxcywh',
                logit_fmt='sigmoid') -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.keep_topk = keep_topk
        self.box_fmt = box_fmt.lower()
        self.logit_fmt = logit_fmt.lower()
        self.logit_func = getattr(F, self.logit_fmt, None)
        self.deploy_mode = False 
    
    def forward(self, outputs: Dict[str, Tensor], orig_target_sizes: Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        pred_boxes = torchvision.ops.box_convert(boxes, in_fmt=self.box_fmt, out_fmt='xyxy') 
        pred_boxes *= torch.tensor(orig_target_sizes, device='cuda').repeat(1, 2).unsqueeze(1)

        values, pred_labels = torch.max(logits, dim=-1)
        
        if self.logit_func:
            pred_scores = self.logit_func(values)
        else:
            pred_scores = values

        # TODO for onnx export
        if self.deploy_mode:
            score_keep = pred_scores[0] > self.score_threshold
            pred_box = pred_boxes[0][score_keep]
            pred_label = pred_labels[0][score_keep]
            pred_score = pred_scores[0][score_keep]

            iou_matrix = box_iou(pred_box, pred_box)
            mask = torch.triu(torch.ones_like(iou_matrix), diagonal=0) # Upper triangular mask including the diagonal elements
            masked_iou_matrix = iou_matrix * (1 - mask)
            indices = torch.nonzero(masked_iou_matrix > 0.9)

            ind_remove = list(set([ind1.item() if pred_score[ind1] < pred_score[ind2] else ind2.item() for (ind1, ind2) in indices]))

            mask = torch.ones(pred_box.size(0), dtype=torch.bool)
            mask[ind_remove] = False

            pred_box = pred_box[mask]
            pred_label = pred_label[mask]
            pred_score = pred_score[mask]

            keep = torchvision.ops.batched_nms(pred_box, pred_score, pred_label, self.iou_threshold)
            keep = keep[:self.keep_topk]

            return pred_label[keep], pred_box[keep], pred_score[keep]

        results = []
        for i in range(logits.shape[0]):
            score_keep = pred_scores[i] > self.score_threshold
            pred_box = pred_boxes[i][score_keep]
            pred_label = pred_labels[i][score_keep]
            pred_score = pred_scores[i][score_keep]

            keep = torchvision.ops.batched_nms(pred_box, pred_score, pred_label, self.iou_threshold)            
            keep = keep[:self.keep_topk]

            blob = {
                'labels': pred_label[keep],
                'boxes': pred_box[keep],
                'scores': pred_score[keep],
            }

            results.append(blob)
            
        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
