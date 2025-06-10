import torch
import torch.nn as nn
import timm


class SwinTImageClassifier(nn.Module):
    def __init__(self, args):
        super(SwinTImageClassifier, self).__init__()
        self.backbone = timm.create_model(
            model_name=args.model_name,
            pretrained=args.pretrained,
            num_classes=0  # backbone만 사용
        )
        self.backbone_out = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_out, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, args.num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        out = self.classifier(x)
        return out
