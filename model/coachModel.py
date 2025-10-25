import torch.nn as nn
from transformers import AutoModel
import torch

class GoKartCoachModel(nn.Module):
    def __init__(self, model_nam, feature_dim = 1024):
        super().__init__()

        # Load frozen DinoV3-ViT-L/16
        #quick explanation as to why we're using this model.
        #Because I only have a 5090 for training and such a small
        #amount of data, I didn't seem necessary to train a larger model,
        #although in the future this might change,
        # especially when we have more data and understand more deeply the current approach. :3

        self.dinov3 = AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m"
        )
        for param in self.dinov3.parameters():
            param.requires_grad = False

        #The chimera has many heads >:3

        # Now the multi head architecture I researched
        # Head 1: Segment type (curve/straight/...)
        self.segment_type_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3)
        )

        # Head 2: Curve number 1-14
        self.curve_number_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 14)
        )

        # Head 3: Direction (left/right/unk)
        self.direction_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
        # Head 4: Racing pints type (Turn_in/Apex/Exit/None)
        self.point_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)
        )

        self.coord_regressor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, pixel_values):
        # Extract features from dinov3
        with torch.no_grad():
            outputs = self.dinov3(pixel_values)
            features = outputs.last_hidden_state[:, 0] #CLS token
        # Multi-head predictions
        segment_type_logits = self.segment_type_head(features)
        curve_number_logits = self.curve_number_head(features)
        direction_logits = self.direction_head(features)
        point_logits = self.point_classifier(features)
        coords = self.coord_regressor(features)

        return segment_type_logits, curve_number_logits, direction_logits, point_logits, coords