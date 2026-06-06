import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split


class ENoseDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        val_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42
    ):
        """
        root_dir: processed_data/
        split: 'train', 'val', or 'test'
        """

        assert split in {"train", "val", "test"}

        self.root_dir = Path(root_dir)
        self.split = split

        # ---------- Step 1: label mapping ----------
        orig_dir = self.root_dir / "orig"
        self.class_names = sorted(
            [d.name for d in orig_dir.iterdir() if d.is_dir()]
        )

        self.label_map = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        # ---------- Step 2: collect original samples ----------
        samples_by_class = {}

        for class_name in self.class_names:
            class_dir = orig_dir / class_name
            files = sorted(class_dir.glob("*.npy"))
            samples_by_class[class_name] = files

        # ---------- Step 3: split based on orig only ----------
        train_samples = []
        val_samples = []
        test_samples = []

        for class_name, files in samples_by_class.items():
            assert len(files) >= 5, f"Not enough samples for class {class_name}"

            train_files = files[:3]
            val_files = files[3:4]
            test_files = files[4:5]

            train_samples.extend([(class_name, f.name) for f in train_files])
            val_samples.extend([(class_name, f.name) for f in val_files])
            test_samples.extend([(class_name, f.name) for f in test_files])

        if split == "train":
            base_samples = train_samples
        elif split == "val":
            base_samples = val_samples
        else:
            base_samples = test_samples

        # ---------- Step 4: expand samples for training ----------
        self.samples = []

        if split == "train":
            aug_dirs = ["orig", "jitter", "scaling", "time_warp", "drift"]
        else:
            aug_dirs = ["orig"]

        for class_name, file_name in base_samples:
            for aug in aug_dirs:
                npy_path = self.root_dir / aug / class_name / file_name
                self.samples.append((npy_path, class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_name = self.samples[idx]

        x = np.load(path)  # (9, 32, 8)
        x = torch.tensor(x, dtype=torch.float32)

        y = self.label_map[class_name]
        y = torch.tensor(y, dtype=torch.long)

        return x, y


