import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from models.ops.modules import MSDeformAttn

class VisualEncoder(nn.Module):
    def __init__(self, 
                d_model=256, 
                nhead=2,
                num_encoder_layers=6, 
                dim_feedforward=1024, 
                dropout=0.1,
                activation="relu", 
                num_feature_scales=4, 
                enc_n_points=4,
                ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_scales = num_feature_scales
        

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_scales , 
                                                          nhead, enc_n_points)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_scales, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, D, H, W = mask.shape
         
        valid_D = torch.sum(~mask[:, :, 0, 0], 1)
        valid_H = torch.sum(~mask[:, 0, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, 0, :], 1)
        valid_ratio_d = valid_D.float() / D
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h, valid_ratio_d], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        masks = []
        for src in srcs:
            masks.append(torch.zeros_like(src[:,:,0,:,:,:]).type(torch.bool))

        
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, nf, c, d, h, w = src.shape
            spatial_shape = (d, h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(3).transpose(2, 3)
            mask = mask.flatten(2)
            pos_embed = pos_embed.flatten(3).transpose(2, 3)
             
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, 1, -1)
            
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        
        src_flatten = torch.cat(src_flatten, 2)
        mask_flatten = torch.cat(mask_flatten, 2)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m[:,0]) for m in masks], 1)

        memory = self.encoder(
            src_flatten, 
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten, 
            mask_flatten
            )
   
        return memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten


class VisualDecoder(nn.Module):
    def __init__(self, 
                d_model=256, 
                nhead=2,
                num_decoder_layers=6, 
                dim_feedforward=1024, 
                dropout=0.1,
                activation="relu", 
                num_feature_scales=4, 
                dec_n_points=4,
                ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_scales = num_feature_scales
        

        decoder_layer = DeformableTransformerDecoderLayer(
                                                        d_model, 
                                                        dim_feedforward,
                                                        dropout, 
                                                        activation,
                                                        num_feature_scales , 
                                                        nhead, 
                                                        dec_n_points)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers)

        self.reference_points = nn.Linear(d_model, 3)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)

    def forward(self, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten, query_embed):
        
       
        bs, t,  _, c = memory.shape
            
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        reference_points = reference_points.unsqueeze(1).repeat(1,t,1,1)     

        hs_cls, hs_loc, inter_references = self.decoder(
                                                query_embed,
                                                tgt,
                                                reference_points,
                                                memory,
                                                spatial_shapes,
                                                level_start_index,
                                                valid_ratios,
                                                mask_flatten,
                                                )
   
        return hs_cls, hs_loc, reference_points, inter_references
   

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=2, n_points=4):
        super().__init__()

        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, 'encode')
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points, 
            src, 
            spatial_shapes, 
            level_start_index, 
            padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
         
        for lvl, (D_, H_, W_) in enumerate(spatial_shapes):

            ref_z, ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
            )
            ref_z = ref_z.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * D_)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y, ref_z), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
         
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)


        for _, layer in enumerate(self.layers):
            output = layer(
                output, 
                pos, 
                reference_points, 
                spatial_shapes, 
                level_start_index, 
                padding_mask)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                d_model=256, 
                d_ffn=1024,
                dropout=0.1, 
                activation="relu",
                n_levels=4, 
                n_heads=8, 
                n_points=4):
        super().__init__()

        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, 'decode')
        self.dropout1_cls = nn.Dropout(dropout)
        self.norm1_cls = nn.LayerNorm(d_model)
        self.dropout1_loc = nn.Dropout(dropout)
        self.norm1_loc = nn.LayerNorm(d_model)

        self.self_attn_cls = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2_cls = nn.Dropout(dropout)
        self.norm2_cls = nn.LayerNorm(d_model)

        self.self_attn_loc = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2_loc = nn.Dropout(dropout)
        self.norm2_loc = nn.LayerNorm(d_model)

        self.linear1_cls = nn.Linear(d_model, d_ffn)
        self.activation_cls = _get_activation_fn(activation)
        self.dropout3_cls = nn.Dropout(dropout)
        self.linear2_cls = nn.Linear(d_ffn, d_model)
        self.dropout4_cls = nn.Dropout(dropout)
        self.norm3_cls = nn.LayerNorm(d_model)

        self.linear1_loc = nn.Linear(d_model, d_ffn)
        self.activation_loc = _get_activation_fn(activation)
        self.dropout3_loc = nn.Dropout(dropout)
        self.linear2_loc = nn.Linear(d_ffn, d_model)
        self.dropout4_loc = nn.Dropout(dropout)
        self.norm3_loc = nn.LayerNorm(d_model)

        self.time_attention_weights = nn.Linear(d_model, 1)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def with_pos_embed_multf(tensor, pos): 
        return tensor if pos is None else tensor + pos.unsqueeze(1)

    def forward_ffn_cls(self, tgt):
        tgt2 = self.linear2_cls(self.dropout3_cls(self.activation_cls(self.linear1_cls(tgt))))
        tgt = tgt + self.dropout4_cls(tgt2)
        tgt = self.norm3_cls(tgt)
        return tgt

    def forward_ffn_loc(self, tgt):
        tgt2 = self.linear2_loc(self.dropout3_loc(self.activation_loc(self.linear1_loc(tgt))))
        tgt = tgt + self.dropout4_loc(tgt2)
        tgt = self.norm3_loc(tgt)
        return tgt

    def forward(
        self, 
        tgt_cls,
        tgt_loc, 
        query_pos, 
        reference_points, 
        src, 
        src_spatial_shapes, 
        level_start_index, 
        src_padding_mask):

        q_cls = self.with_pos_embed(tgt_cls, query_pos)
        k_cls = self.with_pos_embed(tgt_cls, query_pos)

        tgt2_cls = self.self_attn_cls(
            q_cls.transpose(0, 1),
            k_cls.transpose(0, 1),
            tgt_cls.transpose(0, 1))[0].transpose(0, 1)
        tgt_cls = tgt_cls + self.dropout2_cls(tgt2_cls)
        tgt_cls = self.norm2_cls(tgt_cls)        

        if len(tgt_loc.shape) == 3:

            q_loc = self.with_pos_embed(tgt_loc, query_pos)
            k_loc = self.with_pos_embed(tgt_loc, query_pos)
            tgt2_loc = self.self_attn_loc(
                q_loc.transpose(0, 1), 
                k_loc.transpose(0, 1), 
                tgt_loc.transpose(0, 1))[0].transpose(0, 1)
            tgt_loc = tgt_loc + self.dropout2_loc(tgt2_loc)
            tgt_loc = self.norm2_loc(tgt_loc)

            tgt2_loc = self.cross_attn(
                self.with_pos_embed(tgt_loc, query_pos),
                reference_points,
                src, 
                src_spatial_shapes,
                level_start_index,
                src_padding_mask)

        else:
            assert len(tgt_loc.shape) == 4 
            N, nf, num_q,C = tgt_loc.shape
            tgt_list = []
            for i_f in range(nf):  
                tgt_loc_i =  tgt_loc[:,i_f]
                q_loc = self.with_pos_embed(tgt_loc_i, query_pos)
                k_loc = self.with_pos_embed(tgt_loc_i, query_pos)
                tgt2_loc_i = self.self_attn_loc(
                    q_loc.transpose(0, 1), 
                    k_loc.transpose(0, 1), 
                    tgt_loc_i.transpose(0, 1))[0].transpose(0, 1)
                tgt_loc_i = tgt_loc_i + self.dropout2_loc(tgt2_loc_i)
                tgt_loc_i = self.norm2_loc(tgt_loc_i)
                tgt_list.append(tgt_loc_i.unsqueeze(1))
            tgt_loc = torch.cat(tgt_list,dim=1)
            
            tgt2_loc = self.cross_attn(
                self.with_pos_embed_multf(tgt_loc, query_pos),
                reference_points, 
                src, 
                src_spatial_shapes, 
                level_start_index, 
                src_padding_mask)
        
        if len(tgt_loc.shape) == 3: 
            tgt_loc = tgt_loc.unsqueeze(1) + self.dropout1_loc(tgt2_loc)
        else:
            tgt_loc = tgt_loc + self.dropout1_loc(tgt2_loc)
        tgt_loc = self.norm1_loc(tgt_loc)       
        tgt_loc = self.forward_ffn_loc(tgt_loc)

        time_weight = self.time_attention_weights(tgt_loc)
        time_weight = F.softmax(time_weight, 1)

        tgt2_cls = (tgt2_loc*time_weight).sum(1)
        tgt_cls = tgt_cls + self.dropout1_cls(tgt2_cls)
        tgt_cls = self.norm1_cls(tgt_cls)

        tgt_cls = self.forward_ffn_cls(tgt_cls)
        

        return tgt_cls, tgt_loc


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, 
                query_pos, 
                tgt, 
                reference_points, 
                src,
                src_spatial_shapes, 
                src_level_start_index, 
                src_valid_ratios,
                src_padding_mask):

        output_cls = tgt
        output_loc = tgt   

        intermediate_cls = []
        intermediate_loc = []
        intermediate_reference_points = []
        
        for _, layer in enumerate(self.layers):
            
            reference_points_input = reference_points[:, :, :, None] * src_valid_ratios[:,None, None] 
            output_cls, output_loc = layer(
                output_cls,
                output_loc,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes, 
                src_level_start_index, 
                src_padding_mask)
            
            intermediate_cls.append(output_cls)
            intermediate_loc.append(output_loc)
            intermediate_reference_points.append(reference_points)

        return torch.stack(intermediate_cls), torch.stack(intermediate_loc), torch.stack(intermediate_reference_points)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_visual_encoder(args):
    return VisualEncoder(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=0.1,
        activation="relu",
        num_feature_scales=args.num_feature_scales,
        enc_n_points=args.enc_n_points,)

def build_visual_decoder(args):
    return VisualDecoder(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=0.1,
        activation="relu",
        num_feature_scales=args.num_feature_scales,
        dec_n_points=args.dec_n_points
        )


