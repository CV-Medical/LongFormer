import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (accuracy, get_world_size)

from .backbone import build_backbone
from .matcher import build_matcher
from .longformer import Longformer, sigmoid_focal_loss
                           
from .deformable_transformer import build_visual_encoder
from .deformable_transformer import build_visual_decoder
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SetCriterion(nn.Module):

    def __init__(self, args, matcher, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = args.num_classes
        self.classification_type = args.classification_type
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.valid_ratios = None
         

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "instance_labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_labels' in outputs
        src_logits = outputs['pred_labels']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["label"][J] for t, (_, J) in zip(targets, indices)])
        '''300个query的target都是0(NC)或2(sMCI)'''
        if self.classification_type == 'NC/AD':
            base_label = 0
        elif self.classification_type == 'sMCI/pMCI':
            base_label = 2
        target_classes = torch.full(src_logits.shape[:2], base_label, dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
         
        loss_sigmoid_focal = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=0.25, gamma=2) * src_logits.shape[1]
        entropy_loss = nn.CrossEntropyLoss()
        target_classes_o = target_classes_o.cuda()
        loss_ce = entropy_loss(src_logits[idx],target_classes_o)

        losses = {'loss_sigmoid_focal':loss_sigmoid_focal,
                  'loss_ce': loss_ce}

        if log:
            losses['class_error'] = (100 - accuracy(src_logits[idx], target_classes_o)[0])/100.0
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        
  
        valid_ratios = self.valid_ratios
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'].permute(0,2,1,3)[idx]
        num_insts,nf = src_boxes.shape[:2]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
       
        tgt_bbox = tgt_bbox.reshape(num_insts,nf,6)
        sizes = [len(v["label"]) for v in targets]

        target_boxes = list(tgt_bbox.split(sizes,dim=0))


        target_boxes = torch.cat([t[i] for t, (_, i) in zip(target_boxes, indices)], dim=0)
        target_boxes = target_boxes.cuda()
        loss_bbox = F.l1_loss(src_boxes.flatten(1,2), target_boxes.flatten(1,2), reduction='none')
        loss_bbox = loss_bbox/nf

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'classification': self.loss_labels,
            'localization': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes,**kwargs)

    def forward(self, outputs, targets, indices_list, valid_ratios):
        
        self.valid_ratios = valid_ratios
        num_boxes = sum(1 for _ in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices_list[-1], num_boxes, **kwargs))

        return losses


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):    

    device = torch.device(args.device)
    backbone = build_backbone(args, args.num_classes)
    visual_encoder = build_visual_encoder(args)
    visual_decoder = build_visual_decoder(args)
    
    model = Longformer(backbone, visual_encoder, visual_decoder, args)

    matcher = build_matcher(args)
    
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_sigmoid_focal': args.cls_loss_coef}
    weight_dict['loss_bbox'] = args.loc_loss_coef
    weight_dict['loss_giou'] = args.loc_loss_coef

    losses = ['classification', 'localization']

    criterion = SetCriterion(args, 
                            matcher, 
                            weight_dict, 
                            losses)
    criterion.to(device)
       

    return model, criterion


