import torch
from transformers import AutoModel
import torch.nn as nn

class GoKartCoachModel(nn.Module):
    def __init__(self, model_nam, feature_dim = 1024):
        super().__init__()

        self.dinov3 = AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m"
        )
        for param in self.dinov3.parameters():
            param.requires_grad = False

        self.segment_type_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3)
        )

        self.curve_number_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 14)
        )

        self.direction_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
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
        with torch.no_grad():
            outputs = self.dinov3(pixel_values)
            features = outputs.last_hidden_state[:, 0]

            segment_type_logits = self.segment_type_head(features)
            curve_number_logits = self.curve_number_head(features)
            direction_logits = self.direction_head(features)
            point_logits = self.point_classifier(features)
            coords = self.coord_regressor(features)

            return segment_type_logits, curve_number_logits, direction_logits, point_logits, coords

model = GoKartCoachModel(model_nam = "facebook/dinov3-vitl16-pretrain-lvd1689m")

model.load_state_dict(torch.load("model/best_model_v2.pth"))
model.eval()
model.cuda()

#create dummy input for tracing

dummy_input = torch.randn(1, 3, 224, 224).cuda() #DinoV3-ViT-L/16 default size

#trade the model
with torch.no_grad():
    traced_model = torch.jit.trace(model, dummy_input)

traced_model.save("model/coach_model.pt")
print("Model exported to model/coach_model.pt")

with torch.no_grad():
    outputs = traced_model(dummy_input)
    print("\nModel output shapes:")
    print(f" Segment type: {outputs[0].shape}")
    print(f" Curve number: {outputs[1].shape}")
    print(f" Direction: {outputs[2].shape}")
    print(f" Point type: {outputs[3].shape}")
    print(f" Coords: {outputs[4].shape}")