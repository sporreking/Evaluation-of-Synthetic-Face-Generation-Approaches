from pathlib import Path

from PIL import Image
import torch


class TorchImageDataset(torch.utils.data.Dataset):
    """
    Torch compatible dataset for images.
    """

    def __init__(self, img_paths: list[Path], transform):
        """
        Constructs a new torch compatible image dataset.

        Args:
            img_paths (list[Path]): A list of paths of all images to represent.
            transform (_type_): The transformations to apply to each image
                before they are fetched.
        """
        super().__init__()
        self._img_paths = img_paths
        self._transform = transform

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        im = Image.open(self._img_paths[idx]).convert("RGB")
        return self._transform(im)
