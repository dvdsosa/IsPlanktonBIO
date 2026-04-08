from typing import Any

import faiss
import torch
import torch.backends.cudnn as cudnn

from data.models.resnet_big import SupConResNet


def load_faiss_index(index_path: str) -> Any:
    """
    Loads a FAISS index from the specified file path.

    Args:
        index_path (str): The absolute or relative path to the FAISS index.

    Returns:
        Any: The loaded FAISS index object.
    """
    return faiss.read_index(index_path)


def load_model(ckpt_path: str, model_name: str = "resnet50timm") -> torch.nn.Module:
    """
    Instantiates the model architecture and applies the pre-trained weights from a checkpoint.

    Args:
        ckpt_path (str): Path to the PyTorch checkpoint file.
        model_name (str): The name of the underlying model architecture.

    Returns:
        torch.nn.Module: The model loaded with checkpoint weights.
    """
    model = SupConResNet(name=model_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]
    # Remove 'module.' if present
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model.eval()
    cudnn.benchmark = True
    return model
