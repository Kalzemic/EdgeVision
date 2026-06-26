import torch
from torch.utils.data import Dataset


class DetectionDataset(Dataset):


    def __init__(self, images, labels, img_size=224, transform=None):

        self.images = images
        self.labels = labels
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            if len(label) > 0:
                boxes = label[:, 1:].tolist()
                class_labels = label[:, 0].tolist()
            else:
                boxes = []
                class_labels = []
            out = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = out['image']
            if out['bboxes']:
                label = torch.tensor(
                    [[c, *b] for c, b in zip(out['class_labels'], out['bboxes'])],
                    dtype=torch.float32
                )
            else:
                label = torch.zeros((0, 5), dtype=torch.float32)

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        
        if len(label) > 0:
            cls = label[:, 0].long()            
            cx = label[:, 1] * self.img_size
            cy = label[:, 2] * self.img_size
            w = label[:, 3] * self.img_size
            h = label[:, 4] * self.img_size
            boxes = torch.stack([
                cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            ], dim=1).clamp(min=0, max=self.img_size)
        else:
            cls = torch.zeros((0,), dtype=torch.int64)
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            'boxes': boxes,
            'labels': cls,
        }
        return image, target


def Collate_fn(batch):

    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, list(targets)
