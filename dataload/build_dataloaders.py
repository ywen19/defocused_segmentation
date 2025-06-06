from torch.utils.data import DataLoader
from dataload.data_pipe import split_csv, build_iterable_datapipe

def build_dataloaders(
    csv_path: str,
    resize_to=(736, 1280),
    batch_size=4,
    num_workers=4,
    split_ratio=0.8,
    seed=42,
    epoch_seed=None,
    shuffle=True,
    sample_fraction=1.0,
    do_crop: bool = False,
    crop_size=(512, 512)
):
    """
    Build training and validation DataLoaders using TorchData IterableDataPipes.

    Args:
        csv_path (str): Path to the CSV file containing rgb, gt, init_mask columns.
        resize_to (tuple): Resize all images to this resolution (H, W).
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        split_ratio (float): Ratio of training set split (default 0.8 = 80% train, 20% val).
        seed (int): Random seed for data splitting.
        epoch_seed (int): Epoch-level seed for deterministic shuffling.
        shuffle (bool): Whether to shuffle training samples.
        sample_fraction (float): Fraction of data to keep for quick experiments.

    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """

    # Step 1: Split data
    train_rows, val_rows = split_csv(csv_path, split_ratio=split_ratio, seed=seed)
    print(f"val rows: {len(val_rows)}")

    # Step 2: Apply sampling
    train_rows = train_rows[:max(1, int(len(train_rows) * sample_fraction))]
    val_rows = val_rows[:max(1, int(len(val_rows) * sample_fraction))]
    print(f"val rows: {len(val_rows)}")

    # Step 3: Build Iterable DataPipes with epoch seed
    train_pipe = build_iterable_datapipe(
        train_rows, resize_to=resize_to, shuffle=shuffle, seed=epoch_seed or seed,
        do_crop = do_crop, crop_size=crop_size,
        )
    val_pipe = build_iterable_datapipe(
        val_rows, resize_to=resize_to, shuffle=False, seed=seed,
        do_crop = False, crop_size=crop_size,
        )

    # Step 4: Wrap in DataLoader
    train_loader = DataLoader(train_pipe, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_pipe, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
