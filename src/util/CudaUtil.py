import torch


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        print("Using GPU!")
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def empty_cache():
    """Empty the GPU cache"""
    torch.cuda.empty_cache()
