import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from createDataset import GoKartDataset
from model.coachModel import GoKartCoachModel

# setup
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
dataset = GoKartDataset("data/annotations/default.json", "data/annotations", processor)

# Train/val split (80/20)
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# Initialize model
model = GoKartCoachModel("facebook/dinov3-vitl16-pretrain-lvd1689m").cuda()
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Loss function
ce_loss = CrossEntropyLoss()
mse_loss = MSELoss()

#training
num_epochs = 50
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        pixel_values = batch["pixel_val"].cuda()
        segment_type = batch["segment_type"].cuda()
        curve_number = batch["curve_number"].cuda()
        direction = batch["direction"].cuda()
        point_label = batch["point_label"].cuda()
        coords = batch["coords"].cuda()

        #forward pass
        seg_pred, curve_pred, dir_pred, point_pred, coord_pred = model(pixel_values)

        #multi-task loss
        loss_seg = ce_loss(seg_pred, segment_type)
        loss_corner = ce_loss(curve_pred, curve_number)
        loss_dir = ce_loss(dir_pred, direction)
        loss_point = ce_loss(point_pred, point_label)
        loss_coord = mse_loss(coord_pred, coords)

        # weighted combinations
        loss = loss_seg + loss_corner + loss_dir + loss_point + 0.5 * loss_coord

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct_pnt = 0
    correct_curve = 0
    total = 0

    # I know this is the validation loop which duplicated the training one. Its 12:42 am so Ill work on fixing this tmr

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_val"].cuda()
            segment_type = batch["segment_type"].cuda()
            curve_number = batch["curve_number"].cuda()
            direction = batch["direction"].cuda()
            point_label = batch["point_label"].cuda()
            coords = batch["coords"].cuda()

            seg_pred, curve_pred, dir_pred, point_pred, coord_pred = model(pixel_values)

            # Calculate loss
            loss_seg = ce_loss(seg_pred, segment_type)
            loss_corner = ce_loss(curve_pred, curve_number)
            loss_dir = ce_loss(dir_pred, direction)
            loss_point = ce_loss(point_pred, point_label)
            loss_coord = mse_loss(coord_pred, coords)

            loss = loss_seg + loss_corner + loss_dir + loss_point + 0.5 * loss_coord
            val_loss += loss.item()

            correct_pnt += (torch.argmax(point_pred, dim=1) == point_label).sum().item()
            correct_curve += (torch.argmax(curve_pred, dim=1) == curve_number).sum().item()
            total += point_label.size(0)

    point_acc = 100 * correct_pnt / total
    curve_acc = 100 * correct_curve / total
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f" Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Point Acc: {point_acc:.2f}% | Curve Acc: {curve_acc:.2f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"EPOCH DONE GO TO THE NEXT ONE")

print("My misery will be mine, and only mine!")
