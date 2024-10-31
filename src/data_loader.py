import logging
import os
import urllib
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


class ModelNetDownloader:
    dataset_url = "https://modelnet.cs.princeton.edu/ModelNet40.zip"

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def download_and_extract(self) -> None:
        """Download and extract ModelNet40 dataset."""
        # Create root directory if it doesn't exist
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # Download zip file
        zip_path = self.root_dir / "ModelNet40.zip"
        if not zip_path.exists():
            logging.info("Downloading ModelNet40 dataset...")
            urllib.request.urlretrieve(self.dataset_url, zip_path)

        # Extract zip file
        if not (self.root_dir / "ModelNet40").exists():
            logging.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.root_dir)

        logging.info(f"Dataset ready at: {self.root_dir / 'ModelNet40'}")


class ModelNetDataset(data.Dataset):
    """Dataset loader for ModelNet40/10 dataset."""

    def __init__(self, root_dir: str, split: str = "train", num_points: int = 1024, normalize: bool = True):
        """
        Args:
            root_dir (str): Path to the dataset
            split (str): 'train' or 'test'
            num_points (int): Number of points to sample
            normalize (bool): Whether to normalize point clouds
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_points = num_points
        self.normalize = normalize

        # Load all paths and labels
        self.paths, self.labels = self._load_data()

        # Create category to index mapping
        self.categories = sorted(list(set(self.labels)))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

    def _load_data(self) -> tuple[list[Path], list[str]]:
        """Load all file paths and corresponding labels."""
        paths = []
        labels = []

        for category in os.listdir(self.root_dir):
            category_path = self.root_dir / category / self.split
            if not category_path.exists():
                continue

            for file in category_path.glob("*.off"):
                paths.append(file)
                labels.append(category)
        return paths, labels

    def _load_off_file(self, path: Path) -> np.ndarray:
        """Load point cloud from .off file."""
        with open(path, "r") as f:
            lines = f.readlines()

            # Skip header
            if lines[0].strip() == "OFF":
                lines = lines[1:]
                n_verts, n_faces, _ = map(int, lines[0].strip().split())
            elif "OFF" in lines[0].strip():
                n_verts, n_faces, _ = map(int, lines[0][3:].strip().split())

            # Read vertices
            vertices = []
            for i in range(n_verts):
                # logging.info(f"File: {path}. - Line Value: {lines[i + 1]}")
                x, y, z = map(float, lines[i + 1].strip().split())
                vertices.append([x, y, z])

        return np.array(vertices, dtype=np.float32)

    def _normalize_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Normalize point cloud to unit sphere."""
        centroid = np.mean(points, axis=0)
        points -= centroid
        dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points /= dist
        return points

    def _random_sample(self, points: np.ndarray) -> np.ndarray:
        """Randomly sample points if necessary."""
        if len(points) >= self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(points), self.num_points, replace=True)
        return points[indices]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> list[torch.Tensor, int]:
        """Get a single point cloud and its label."""
        path = self.paths[idx]
        category = self.labels[idx]

        # Load and preprocess point cloud
        points = self._load_off_file(path)
        points = self._random_sample(points)

        if self.normalize:
            points = self._normalize_point_cloud(points)

        # Convert to tensor and get label index
        points = torch.from_numpy(points).float()
        label = self.category_to_idx[category]

        return points, label


class DataModule:
    """Data module for handling dataset loading and preparation."""

    def __init__(self, data_dir: str, batch_size: int = 32, num_points: int = 1024, num_workers: int = 4):
        """
        Args:
            data_dir (str): Path to dataset
            batch_size (int): Batch size for dataloaders
            num_points (int): Number of points per point cloud
            num_workers (int): Number of workers for data loading
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_points = num_points
        self.num_workers = num_workers

    def setup(self) -> None:
        """Set up train and test datasets."""
        self.train_dataset = ModelNetDataset(self.data_dir, split="train", num_points=self.num_points)
        self.test_dataset = ModelNetDataset(self.data_dir, split="test", num_points=self.num_points)

    def train_dataloader(self) -> data.DataLoader:
        """Get training data loader."""
        return data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self) -> data.DataLoader:
        """Get test data loader."""
        return data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )


# Complete setup
if __name__ == "__main__":
    # Download and prepare dataset
    downloader = ModelNetDownloader("data")
    downloader.download_and_extract()

    # Update the DataModule to use the correct path
    data_module = DataModule(data_dir="data/ModelNet40", batch_size=32, num_points=1024, num_workers=1)

    # Setup datasets
    data_module.setup()

    # Get data loaders
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    # Verify data loading
    for points, labels in train_loader:
        logging.info(f"Batch points shape: {points.shape}")
        logging.info(f"Batch labels shape: {labels.shape}")
        break
