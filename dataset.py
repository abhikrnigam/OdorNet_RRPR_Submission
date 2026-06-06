import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ENoseDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
    ):
        """
        root_dir: OdorNet_Dataset/
        split: 'train', 'val', or 'test'

        Dataset structure:
            root_dir/
                <class_name>/
                    <class_name>_cycle_0.npy
                    ...
                    <class_name>_cycle_24.npy
        Each .npy file has shape (9, 32, 8).
        25 cycles per class across 50 classes = 1,250 samples total.
        """

        assert split in {"train", "val", "test"}

        self.root_dir = Path(root_dir)
        self.split = split

        # ---------- Step 1: label mapping ----------
        self.class_names = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        self.label_map = {name: idx for idx, name in enumerate(self.class_names)}

        # ---------- Step 2: assign cycles to splits ----------
        # Cycles are grouped in blocks of 5 (5 recordings per session × 5 sessions).
        # Within each block: indices 0,1,2 → train | index 3 → val | index 4 → test
        # Train uses the corresponding indices across all 5 blocks (15 samples/class).
        # Val and test use only the first block (1 sample/class each).

        BLOCK = 5
        NUM_BLOCKS = 5

        train_indices = [b * BLOCK + i for b in range(NUM_BLOCKS) for i in range(3)]  # 15 per class
        val_indices   = [3]   # 1 per class
        test_indices  = [4]   # 1 per class

        if split == "train":
            cycle_indices = train_indices
        elif split == "val":
            cycle_indices = val_indices
        else:
            cycle_indices = test_indices

        # ---------- Step 3: collect samples ----------
        self.samples = []

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            # Sort files numerically by cycle number
            files = sorted(
                class_dir.glob("*.npy"),
                key=lambda p: int(p.stem.split("_cycle_")[1])
            )
            assert len(files) == 25, f"Expected 25 cycles for {class_name}, found {len(files)}"

            for idx in cycle_indices:
                self.samples.append((files[idx], class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_name = self.samples[idx]

        x = np.load(path)  # (9, 32, 8)
        x = torch.tensor(x, dtype=torch.float32)

        y = self.label_map[class_name]
        y = torch.tensor(y, dtype=torch.long)

        return x, y
