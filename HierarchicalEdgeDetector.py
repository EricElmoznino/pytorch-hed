import math
import torch
from torch import nn
from torch.nn import functional as F


class HierarchicalEdgeDetector(nn.Module):

    def __init__(self, saved_model_path='bsds500.pth'):
        super().__init__()

        self.netVggOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netVggTwo = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netVggThr = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netVggFou = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netVggFiv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netScoreOne = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.dataset_mean = None
        self.dataset_scale = None
        self.model_size = None
        if saved_model_path is not None:
            self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                  torch.load(saved_model_path, map_location=lambda storage, loc: storage).items()})
            self.eval()
            if 'bsds500.pth' in saved_model_path:
                self.dataset_mean = torch.FloatTensor([122.67891434, 116.66876762, 104.00698793]).view(1, 3, 1, 1)
                self.dataset_scale = torch.FloatTensor([255, 255, 255]).view(1, 3, 1, 1)
                self.model_size = (320, 480)

    def forward(self, image):
        image, pad = self.resize_to_model(image)
        image = self.normalize(image)

        tenVggOne = self.netVggOne(image)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = F.interpolate(input=tenScoreOne, size=(image.shape[2], image.shape[3]), mode='bilinear')
        tenScoreTwo = F.interpolate(input=tenScoreTwo, size=(image.shape[2], image.shape[3]), mode='bilinear')
        tenScoreThr = F.interpolate(input=tenScoreThr, size=(image.shape[2], image.shape[3]), mode='bilinear')
        tenScoreFou = F.interpolate(input=tenScoreFou, size=(image.shape[2], image.shape[3]), mode='bilinear')
        tenScoreFiv = F.interpolate(input=tenScoreFiv, size=(image.shape[2], image.shape[3]), mode='bilinear')

        edges = self.netCombine(torch.cat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1))
        edges = self.crop_pad(edges, pad)
        return edges

    def normalize(self, image):
        if self.dataset_scale is not None:
            self.dataset_scale = self.dataset_scale.to(image.device)
            image *= self.dataset_scale
        if self.dataset_mean is not None:
            self.dataset_mean = self.dataset_mean.to(image.device)
            image -= self.dataset_mean
        return image

    def resize_to_model(self, image):
        orig_size = tuple(image.shape[-2:])
        if self.model_size is None or orig_size == self.model_size:
            return image, [0, 0, 0, 0]

        target_ar = self.model_size[0] / self.model_size[1]
        ar = image.shape[2] / image.shape[3]
        if ar > target_ar:      # height too tall, will end up padding width
            image = F.interpolate(image, (self.model_size[0], int(self.model_size[0] / ar)), mode='bilinear')
        else:                   # width too large, will end up padding height
            image = F.interpolate(image, (int(self.model_size[1] * ar), self.model_size[1]), mode='bilinear')

        dif = (self.model_size[0] - image.shape[2], self.model_size[1] - image.shape[3])
        pad = [math.floor(dif[1] / 2), math.ceil(dif[1] / 2), math.floor(dif[0] / 2), math.ceil(dif[0] / 2)]
        image = F.pad(image, pad, value=1.0)

        return image, pad

    def crop_pad(self, image, pad):
        if all([p == 0 for p in pad]):
            return image
        height_bounds = (pad[2], image.size(2) - pad[3])
        width_bounds = (pad[0], image.size(3) - pad[1])
        image = image[:, :, height_bounds[0]:height_bounds[1], width_bounds[0]:width_bounds[1]]
        return image

