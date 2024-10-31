import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from src.data_loader import DataModule


class TNet(nn.Module):
    """Input/Feature Transform Net (T-Net)"""

    def __init__(self, k: int):
        """
        Args:
            k (int): Dimension of the transform matrix (3 for input, 64 for feature)
        """
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as identity matrix
        identity = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1)
        if x.is_cuda:
            identity = identity.cuda()

        x = x.view(-1, self.k, self.k) + identity
        return x


class PointNet(nn.Module):
    """PointNet Architecture"""

    def __init__(self, num_classes: int = 40, dropout: float = 0.3):
        super().__init__()

        # Input transform (3x3)
        self.input_transform = TNet(k=3)

        # First MLP
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        # Feature transform (64x64)
        self.feature_transform = TNet(k=64)

        # Second MLP
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        # Global feature MLP
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> tuple:
        num_points = x.size(2)

        # Input transform
        trans = self.input_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), trans).transpose(1, 2)

        # First MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transform
        trans_feat = self.feature_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), trans_feat).transpose(1, 2)

        # Second MLP
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # MLP for global features
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetLoss(nn.Module):
    """Custom PointNet loss with regularization"""

    def __init__(self, mat_diff_loss_scale: float = 0.001):
        super().__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor, trans_feat: torch.Tensor) -> torch.Tensor:
        loss = F.nll_loss(pred, target)

        # Regularization loss for feature transform matrix
        d = trans_feat.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans_feat.is_cuda:
            I = I.cuda()
        mat_diff = torch.mean(torch.norm(torch.bmm(trans_feat, trans_feat.transpose(2, 1)) - I, dim=(1, 2)))

        total_loss = loss + mat_diff * self.mat_diff_loss_scale
        return total_loss


class PointNetTrainer:
    """Handles training and evaluation of PointNet model"""

    def __init__(self, model: nn.Module, data_module: DataModule, device: str = "cuda", lr: float = 0.001):
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.criterion = PointNetLoss()

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0

        for points, target in tqdm(self.data_module.train_dataloader()):
            points = points.transpose(1, 2).to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            pred, trans, trans_feat = self.model(points)
            loss = self.criterion(pred, target, trans_feat)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.data_module.train_dataloader())

    def evaluate(self) -> tuple:
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for points, target in self.data_module.test_dataloader():
                points = points.transpose(1, 2).to(self.device)
                target = target.to(self.device)

                pred, _, _ = self.model(points)
                pred_choice = pred.max(1)[1]
                correct = pred_choice.eq(target).sum().item()

                total_correct += correct
                total_samples += points.size(0)

        accuracy = total_correct / total_samples
        return accuracy


# Training script
def train_pointnet():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = DataModule(data_dir="data/ModelNet40", batch_size=32, num_points=1024, num_workers=0)
    data_module.setup()

    model = PointNet(num_classes=40)
    trainer = PointNetTrainer(model, data_module, device=device)

    # Training loop
    num_epochs = 100
    best_accuracy = 0

    for epoch in range(num_epochs):
        loss = trainer.train_epoch()
        accuracy = trainer.evaluate()

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_pointnet.pth")
