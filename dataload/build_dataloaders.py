from torch.utils.data import DataLoader
from data_pipe import split_csv, build_iterable_datapipe

def build_dataloaders(
    csv_path: str,
    resize_to=(720, 1280),
    batch_size=4,
    num_workers=4,
    split_ratio=0.8,
    seed=42,
    shuffle=True
):
    """
    Build training and validation DataLoaders using TorchData IterableDataPipes.

    Args:
        csv_path (str): Path to the CSV file containing rgb, gt, init_mask columns.
        resize_to (tuple): Resize all images to this resolution.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        split_ratio (float): Ratio of training set split (default 0.8 = 80% train, 20% val).
        seed (int): Random seed for reproducibility.
        shuffle (bool): Whether to shuffle training samples.

    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    # Split CSV into train and val sample lists
    train_rows, val_rows = split_csv(csv_path, split_ratio=split_ratio, seed=seed)

    # Build separate iterable pipelines
    train_pipe = build_iterable_datapipe(train_rows, resize_to=resize_to, shuffle=shuffle)
    val_pipe = build_iterable_datapipe(val_rows, resize_to=resize_to, shuffle=False)

    # Wrap in DataLoader
    train_loader = DataLoader(train_pipe, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_pipe, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
