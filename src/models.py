from typing import Any

import faiss
import torch
import torch.backends.cudnn as cudnn

from data.models.resnet_big import SupConResNet


def load_faiss_index(index_path: str) -> Any:
    """
    Loads a FAISS indexing structure from the given file path.

    Args:
        index_path (str): The file path to the saved FAISS index binary.

    Returns:
        Any: The instantiated FAISS index object used for similarity searching.
    """
    return faiss.read_index(index_path)


def load_model(ckpt_path: str, model_name: str = "resnet50timm") -> torch.nn.Module:
    """
    Instantiates the Convolutional Neural Network and loads its weights from a checkpoint.

    This function adjusts module prefixes in the state dictionary (if saved via DataParallel)
    and places the loaded model in evaluation mode onto the available GPU.

    Args:
        ckpt_path (str): The file path to the model weights.
        model_name (str, optional): The name corresponding to the architecture. Defaults to 'resnet50timm'.

    Returns:
        torch.nn.Module: The loaded neural network model ready for inference.
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
