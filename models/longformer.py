import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from util.misc import inverse_sigmoid


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class Longformer(nn.Module):
    def __init__(self, backbone, visual_encoder, visual_decoder, args):
        super().__init__()
        self.backbone = backbone
        self.visual_encoder = visual_encoder
        self.visual_decoder = visual_decoder
        self.num_visits = args.num_visits
        self.num_queries = args.num_queries
        self.hidden_dim = hidden_dim = args.hidden_dim

        
        input_proj_list = []
        input_proj_list.append(nn.Sequential(
                    nn.Conv3d(64, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
        input_proj_list.append(nn.Sequential(
                    nn.Conv3d(80, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
        input_proj_list.append(nn.Sequential(
                    nn.Conv3d(96, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
        input_proj_list.append(nn.Sequential(
                    nn.Conv3d(128, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
        self.input_proj = nn.ModuleList(input_proj_list)

        self.num_feature_scales = 4
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim*2)

        num_classes = args.num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        dec_layer_num = visual_decoder.decoder.num_layers
        self.class_embed = _get_clones(self.class_embed, dec_layer_num)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        self.bbox_embed = _get_clones(self.bbox_embed, dec_layer_num)
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[3:], -3.0)

    def forward(self, samples, gt_targets, criterion, train=False):
        '''step1: backbone'''

        features, pos = self.backbone(samples)
        srcs = []
        poses = []

        for l, feat in enumerate(features):
            src_proj_l = self.input_proj[l](feat)
            
            n, c, d, h, w = src_proj_l.shape
            src_proj_l = src_proj_l.reshape(n//self.num_visits, self.num_visits, c, d, h, w)
            
            np, cp, dp, hp, wp = pos[l].shape
            pos_l = pos[l].reshape(np//self.num_visits, self.num_visits, cp, dp, hp, wp)
            
            srcs.append(src_proj_l)
            poses.append(pos_l)

        memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten = self.visual_encoder(srcs, poses)

        query_embeds = self.query_embed.weight
        hs_cls, hs_loc, init_reference, inter_references = self.visual_decoder(
            memory, 
            spatial_shapes, 
            level_start_index, 
            valid_ratios, 
            mask_flatten, 
            query_embeds)
        
        
        valid_ratios = valid_ratios[:,0]

        outputs = {}
        outputs_classes = []
        outputs_coords = []
        indices_list = []

        dec_lay_num = hs_cls.shape[0]

        for lvl in range(dec_lay_num):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.class_embed[lvl](hs_cls[lvl])

            tmp = self.bbox_embed[lvl](hs_loc[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 3
                tmp[..., :3] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_layer = {'pred_labels': outputs_class, 'pred_boxes': outputs_coord}

            indices = criterion.matcher(outputs_layer, gt_targets, self.num_visits)
            indices_list.append(indices)

        outputs['pred_labels'] = outputs_classes[-1]
        outputs['pred_boxes'] = outputs_coords[-1]

        loss_dict = criterion(outputs, gt_targets, indices_list, valid_ratios)
        

        return outputs, loss_dict, indices_list



def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    input: bs, num_query, num_class 
    targets: bs, num_query, num_class 
    alpha:  0.25
    gamma: 2
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
