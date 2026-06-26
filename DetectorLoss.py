import torch
import torch.nn as nn
from torchvision.ops import complete_box_iou_loss


class DetectorLoss(nn.Module):
    def __init__(self, B=2, S=7, lambda_coord=5, lambda_noobj=.5, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.B = B
        self.S = S
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, pred, target):
        batchsize = pred.shape[0]

        pred = pred.view(batchsize, self.B, (5 + self.num_classes), self.S, self.S)
        target = target.view(batchsize, self.B, (5 + self.num_classes), self.S, self.S)

        obj_mask_4d = (target[:, :, 4, :, :] > 0)              # (N, B, S, S)
        no_obj_mask_4d = ~obj_mask_4d

        obj_idx = obj_mask_4d.nonzero()                        # (M, 4): batch, b, i, j
        i = obj_idx[:, 2]
        j = obj_idx[:, 3]

        pred_box = pred[:, :, 0:4, :, :].permute(0, 1, 3, 4, 2)[obj_mask_4d]   # (M, 4)
        tgt_box  = target[:, :, 0:4, :, :].permute(0, 1, 3, 4, 2)[obj_mask_4d]

        px = (j + torch.sigmoid(pred_box[:, 0])) / self.S
        py = (i + torch.sigmoid(pred_box[:, 1])) / self.S
        pw = torch.sigmoid(pred_box[:, 2])
        ph = torch.sigmoid(pred_box[:, 3])

        tx = (j + tgt_box[:, 0]) / self.S
        ty = (i + tgt_box[:, 1]) / self.S
        tw = tgt_box[:, 2]
        th = tgt_box[:, 3]

        pred_corners = torch.stack([px - pw/2, py - ph/2, px + pw/2, py + ph/2], dim=1)
        tgt_corners  = torch.stack([tx - tw/2, ty - th/2, tx + tw/2, ty + th/2], dim=1)

        if obj_mask_4d.any():
            coord_loss = self.lambda_coord * complete_box_iou_loss(
                pred_corners, tgt_corners, reduction='sum'
            )
        else:
            coord_loss = torch.tensor(0.0, device=pred.device)

        conf_loss_obj = self.bce(pred[:, :, 4, :, :][obj_mask_4d],
                                 target[:, :, 4, :, :][obj_mask_4d])
        conf_loss_noobj = self.lambda_noobj * self.bce(
            pred[:, :, 4, :, :][no_obj_mask_4d],
            target[:, :, 4, :, :][no_obj_mask_4d]
        )

        pred_cls = pred[:, :, 5:, :, :].permute(0, 1, 3, 4, 2)[obj_mask_4d]
        tgt_cls  = target[:, :, 5:, :, :].permute(0, 1, 3, 4, 2)[obj_mask_4d].argmax(dim=-1)

        if obj_mask_4d.any():
            cls_loss = self.ce(pred_cls, tgt_cls)
        else:
            cls_loss = torch.tensor(0.0, device=pred.device)

        total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + cls_loss
        return total_loss / (batchsize * self.S * self.S * self.B)