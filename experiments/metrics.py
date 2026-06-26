import torch



def decode_predictions(pred, S, B, C, conf_thresh=0.25):
    """
    pred: (N, B*(5+C), S, S) raw head output
    Returns list of N dicts with 'boxes' (xyxy in [0,1]), 'scores', 'labels'.
    """
    N = pred.shape[0]
    pred = pred.view(N, B, 5 + C, S, S).permute(0, 1, 3, 4, 2)  # (N, B, S, S, 5+C)

    device = pred.device
    cell_y = torch.arange(S, device=device).view(1, 1, S, 1).float()
    cell_x = torch.arange(S, device=device).view(1, 1, 1, S).float()

    x = (torch.sigmoid(pred[..., 0]) + cell_x) / S
    y = (torch.sigmoid(pred[..., 1]) + cell_y) / S
    w = torch.sigmoid(pred[..., 2])
    h = torch.sigmoid(pred[..., 3])
    obj = torch.sigmoid(pred[..., 4])              # (N, B, S, S)
    cls_probs = torch.softmax(pred[..., 5:], dim=-1)  # (N, B, S, S, C)
    cls_score, cls_idx = cls_probs.max(dim=-1)     # (N, B, S, S)
    score = obj * cls_score

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (N, B, S, S, 4)

    results = []
    for n in range(N):
        s = score[n].flatten()
        b = boxes[n].reshape(-1, 4)
        l = cls_idx[n].flatten()
        keep = s > conf_thresh
        results.append({'boxes': b[keep], 'scores': s[keep], 'labels': l[keep]})
    return results


def decode_targets(target, S, B, C):
    """target: (N, B, 5+C, S, S) -> list of dicts with 'boxes' and 'labels'."""
    N = target.shape[0]
    target = target.permute(0, 1, 3, 4, 2)  # (N, B, S, S, 5+C)
    device = target.device
    cell_y = torch.arange(S, device=device).view(1, 1, S, 1).float()
    cell_x = torch.arange(S, device=device).view(1, 1, 1, S).float()

    obj_mask = target[..., 4] > 0
    x = (target[..., 0] + cell_x) / S
    y = (target[..., 1] + cell_y) / S
    w = target[..., 2]
    h = target[..., 3]
    cls_idx = target[..., 5:].argmax(dim=-1)

    x1 = x - w / 2; y1 = y - h / 2; x2 = x + w / 2; y2 = y + h / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    results = []
    for n in range(N):
        m = obj_mask[n]
        results.append({'boxes': boxes[n][m], 'labels': cls_idx[n][m]})
    return results