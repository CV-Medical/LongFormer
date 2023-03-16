import torch

def box_area(boxes):
    return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])


def box_cxcyczdwh_to_xyzxyz(x):
    x_c, y_c, z_c, d, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * d), (y_c - 0.5 * w), (z_c - 0.5 * h),
         (x_c + 0.5 * d), (y_c + 0.5 * w), (z_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])

    dwh = (rb - lt).clamp(min=0)
    inter = dwh[:, :, 0] * dwh[:, :, 1] * dwh[:, :, 2]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    dwh = (rb - lt).clamp(min=0)
    area = dwh[:, :, 0] * dwh[:, :, 1] * dwh[:, :, 2]

    return iou - (area - union) / area

