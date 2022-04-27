from pathlib import Path

from PIL import Image
import torch.utils.data
import pandas as pd
import math

from typing import List


class TorchImageDataset(torch.utils.data.Dataset):
    """
    Torch compatible dataset for images.
    """

    def __init__(
        self,
        img_paths: List[Path],
        transform,
        attr: str = None,
        df: pd.DataFrame = None,
    ):
        """
        Constructs a new torch compatible image dataset.

        Args:
            img_paths (list[Path]): A list of paths of all images to represent.
            transform (-): The transformations to apply to each image
                before they are fetched.
            attr (str, optional): Name of the attribute used as labels. Defaults to None.
            df (pd.DataFrame, optional): Dataframe containing the labels. Defaults to None.
        """
        super().__init__()
        self._transform = transform
        self._attr = attr
        self._df = df

        # If created with an attribute, remove unlabeled images.
        self._img_paths = (
            img_paths if attr is None else self._remove_unlabeled_images(img_paths)
        )

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        im = Image.open(self._img_paths[idx]).convert("RGB")
        if self._df is None or self._attr is None:
            return self._transform(im)
        else:
            return self._transform(im), self._get_label(self._img_paths[idx])

    def _parse_index(self, image_filename: str):
        # Parse filename
        # Assumes '.....XXXXX.ext'
        # Where XXXXX is the index
        digits = int(math.log10(max(self._df.index))) + 1
        return int(str(image_filename).split(".")[-2][-digits:])

    def _get_label(self, image_filename: str):
        idx = self._parse_index(image_filename)
        res = self._df.loc[[idx]][self._attr].values[0]
        return res

    def _remove_unlabeled_images(self, image_paths: List[Path]) -> List[Path]:
        indexes = self._df.index
        digits = int(math.log10(max(self._df.index))) + 1
        paths = []

        # Find unlabeled images
        for path in image_paths:
            index = int(str(path).split(".")[-2][-digits:])
            if index in indexes:
                if self._df.loc[[index]][self._attr].isnull().values[0]:
                    continue
                else:
                    paths.append(path)
            else:
                continue

        print(f"Removed {len(image_paths)- len(paths)} unlabeled images!")
        return paths
