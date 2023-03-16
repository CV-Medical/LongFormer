import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcyczdwh_to_xyzxyz, generalized_box_iou


class HungarianMatcher(nn.Module):
   

    def __init__(self,
                 cost_cls: float = 1,
                 cost_loc: float = 1):
        super().__init__()
        self.cost_cls = cost_cls
        self.cost_loc = cost_loc

    def forward(self, outputs, targets, n_visits):

         

        with torch.no_grad():
            bs, num_queries = outputs["pred_labels"].shape[:2]

            out_cls_prob = outputs["pred_labels"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].permute(0,2,1,3).flatten(0, 1)

            tgt_cls = torch.cat([v["label"] for v in targets]).long()
            tgt_bbox = torch.stack([v["boxes"] for v in targets])
            
            tgt_bbox = tgt_bbox.permute(0,2,1,3).flatten(0,1)


            tgt_bbox = tgt_bbox.cuda()
            cost_bbox = torch.cdist(out_bbox.flatten(1,2), tgt_bbox.flatten(1,2))

            cost_giou = 0
            for i in range(n_visits):
                cost_giou += -generalized_box_iou(box_cxcyczdwh_to_xyzxyz(out_bbox[:,i]),
                                                box_cxcyczdwh_to_xyzxyz(tgt_bbox[:,i]))
            cost_giou = cost_giou/n_visits

            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_cls_prob ** gamma) * (-(1 - out_cls_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_cls_prob) ** gamma) * (-(out_cls_prob + 1e-8).log())
            cost_cls = pos_cost_class[:, tgt_cls] - neg_cost_class[:, tgt_cls]

            C = self.cost_loc * cost_bbox + self.cost_cls * cost_cls + self.cost_loc * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            _num = [len(v['label']) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(_num, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), 
                     torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_cls=args.set_cost_cls,
                            cost_loc=args.set_cost_loc)


