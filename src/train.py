from jupyter_client.consoleapp import classes
from pyarrow.dataset import dataset
from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn,FasterRCNN_MobileNet_V3_Large_FPN_Weights
from pprint import pprint
import os
from pprint import pprint
import torch
from torchvision.transforms import ToTensor,Compose
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize

root = '../data'
class CustomeVOC(VOCDetection):
    def __init__(self,root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download, transform)

        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']

    def __getitem__(self, item):
        image, target = super().__getitem__(item)

        bboxes = []
        labels = []

        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            x_max = float(bbox['xmax'])
            x_min = float(bbox['xmin'])
            y_min = float(bbox['ymin'])
            y_max = float(bbox['ymax'])

            class_name = obj['name']

            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.classes.index(class_name))

        final_target = {"boxes": torch.FloatTensor(bboxes),
                        "labels": torch.LongTensor(labels)}
        return image, final_target
def collate_fn(batch):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets
def Train():
    temp = '/kaggle/input/voc2012'

    transform = Compose([
        ToTensor(),
    ])

    train_data = CustomeVOC(temp, year="2012", image_set="train", download=False, transform=transform)
    val_data = CustomeVOC(temp, year="2012", image_set="val", download=False, transform=transform)

    train_loader = DataLoader(
        dataset=train_data,
        num_workers=8,
        shuffle=True,
        drop_last=True,
        batch_size=32,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        dataset=val_data,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        batch_size=32
    )

    # gpu setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        target_list = [{
                'boxes': t['boxes'].to(device),
                'labels': t['labels'].to(device)
           } for t in targets
        ]
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
    model.train()

    for images, targets in train_loader:
        loss = model(images, targets)
        print(loss.item())

if __name__ == '__main__':
    Train()
