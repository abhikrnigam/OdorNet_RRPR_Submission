from torch.utils.data import DataLoader
from dataset import ENoseDataset


def create_dataloaders(
    data_dir="OdorNet_Dataset/",
    batch_size=8,
    num_workers=4
):
    train_ds = ENoseDataset(data_dir, split="train")
    val_ds = ENoseDataset(data_dir, split="val")
    test_ds = ENoseDataset(data_dir, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
