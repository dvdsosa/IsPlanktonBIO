from typing import Any, Tuple

import numpy as np
import torch
from torchvision import datasets, transforms


class CustomDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(CustomDataset, self).__getitem__(index)
        path, _ = self.samples[index]
        class_name = self.classes[original_tuple[1]]
        return original_tuple[0], class_name


def loader_with_paths(loader):
    """
    Yields individual samples from a PyTorch DataLoader along with their file paths.

    Args:
        loader (torch.utils.data.DataLoader): The DataLoader containing the dataset.

    Yields:
        Tuple[torch.Tensor, str, str]: A tuple containing:
            - data_batch (torch.Tensor): A single image tensor with a batch dimension.
            - label (str): The class label for the image.
            - path (str): The file path of the original image.
    """
    loader_iter = iter(loader)
    dataset_samples = iter(loader.dataset.samples)

    try:
        while True:
            # Get next batch from loader
            data_batch, label_batch = next(loader_iter)
            path, _ = next(dataset_samples)

            yield data_batch, label_batch[0], path

    except StopIteration:
        # End of iterator
        return


def set_loader(dataset_root_path: str, transform_cfg: dict = None) -> Tuple[Any, Any]:
    """
    Initializes and returns PyTorch DataLoaders for both intermediate stages of the pipeline.

    Args:
        dataset_root_path (str): The root directory path for the dataset.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: A tuple containing:
            - stage1_loader: DataLoader with stage-1 specific transformations.
            - stage2_loader: DataLoader with stage-2 specific transformations.
    """
    mean_stage1 = (
        transform_cfg["stage1"]["mean"] if transform_cfg else (0.0418, 0.0353, 0.0409)
    )
    std_stage1 = (
        transform_cfg["stage1"]["std"] if transform_cfg else (0.0956, 0.0911, 0.0769)
    )
    normalize_stage1 = transforms.Normalize(mean=mean_stage1, std=std_stage1)
    stage1_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_stage1,
        ]
    )
    stage1_dataset = CustomDataset(root=dataset_root_path, transform=stage1_transform)
    stage1_loader = torch.utils.data.DataLoader(
        stage1_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    stage2_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    stage2_dataset = CustomDataset(root=dataset_root_path, transform=stage2_transform)
    stage2_loader = torch.utils.data.DataLoader(
        stage2_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    return stage1_loader, stage2_loader


def apply_additional_transforms_stage2(
    image_tensor: Any, transform_cfg: dict = None
) -> Any:
    """
    Applies stage-specific torchvision transforms to an input image tensor or NumPy array.

    The function converts the input to a tensor (if necessary), normalizes pixel values
    to the [0, 1] range, resizes the image to 256x256, applies a 224x224 center crop,
    and finally normalizes utilizing stage-2 particular mean and standard deviation values.

    Args:
        image_tensor (Any): Input image representation (torch.Tensor or np.ndarray).

    Returns:
        Any: The transformed image in its original data format (tensor or array).
    """

    original_format = type(image_tensor)
    original_shape_dims = None

    # Convert numpy array to torch tensor and permute to (C, H, W) if needed
    if isinstance(image_tensor, np.ndarray):
        original_shape_dims = image_tensor.ndim
        if image_tensor.ndim == 3:  # (H, W, C)
            image_tensor = (
                torch.from_numpy(image_tensor).permute(2, 0, 1).float() / 255.0
            )
        elif image_tensor.ndim == 2:  # (H, W)
            image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).float() / 255.0
        else:
            raise ValueError("Unsupported numpy array shape for image_tensor.")
    # If input is already a tensor but not float, convert to float
    elif isinstance(image_tensor, torch.Tensor):
        original_shape_dims = image_tensor.dim()
        if not torch.is_floating_point(image_tensor):
            image_tensor = image_tensor.float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

    mean_stage2 = (
        transform_cfg["stage2"]["mean"] if transform_cfg else (0.0425, 0.0436, 0.0432)
    )
    std_stage2 = (
        transform_cfg["stage2"]["std"] if transform_cfg else (0.1466, 0.1488, 0.1476)
    )
    normalize_stage2 = transforms.Normalize(mean=mean_stage2, std=std_stage2)
    additional_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), normalize_stage2]
    )

    # If image_tensor is a batch (B, C, H, W) with batch size 1, remove batch dimension
    if image_tensor.dim() == 4 and image_tensor.size(0) == 1:
        image_tensor = image_tensor.squeeze(0)

    # Apply the additional transforms
    image_tensor = additional_transforms(image_tensor)

    # Return in original format
    if original_format == np.ndarray:
        # Convert back to numpy array
        if original_shape_dims == 3:  # Original was (H, W, C)
            return image_tensor.permute(1, 2, 0).numpy()
        elif original_shape_dims == 2:  # Original was (H, W)
            return image_tensor.squeeze(0).numpy()
    else:
        # Return as tensor in original shape
        if original_shape_dims == 4:  # Original was (B, C, H, W)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
        elif original_shape_dims == 3:  # Original was (C, H, W)
            pass  # Already in correct format

    return image_tensor
