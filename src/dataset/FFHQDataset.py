from src.dataset.Dataset import Dataset
from src.util.FileJar import FileJar

from src.util.ZeroPaddedIterator import zero_padded_iterator

import os, json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
from pathlib import Path


class FFHQDataset(Dataset):
    """
    A subclass to Dataset which provides the ability to extract labels
    from the FFHQ dataset. These labels are saved to disk which allows subsequent
    creations of the dataset to be performed without extraction the labels
    again.

    Before creating a FFQDataset, a directory should
    exist with the name of "FFHQ_`resolution`" in the dataset directory.
    There should also exist appropriate directories for the labels ("label/")
    and images ("image/").
    """

    # Subdirectory name to the images
    DS_DIR_IMAGES = "image/"

    # Subdirectory name to the labels
    DS_DIR_LABELS = "label/"

    # Filename of the extracted labels
    DS_LABELS_NAME = "labels.pkl"

    def __init__(self, resolution: int = 256):
        """
        Constructor which extract labels from the dataset and saves to disk.
        This process is managed by a `FileJar` which checks if
        previous instances of the FFHQDataset have extracted labels before.
        If this is the case extraction is not needed and the result is loaded
        from disk instead.

        The name of the dataset is a combination of FFHQ and the specified resolution.

        Args:
            resolution (int, optional): The native resolution of the dataset.
                Defaults to 256.
        """
        super(FFHQDataset, self).__init__(resolution)

    def get_resolution_invariant_name() -> str:
        return "FFHQ"

    def is_ready(resolution: int) -> bool:
        return (
            Path(Dataset.DS_DIR_PREFIX)
            / FFHQDataset.get_name(resolution)
            / FFHQDataset.DS_LABELS_NAME
        ).is_file()

    def get_image_dir(self) -> Path:
        return self.get_path() / self.DS_DIR_IMAGES

    def get_image_paths(self) -> list[Path]:
        return self._img_paths

    def get_processed_labels(self) -> pd.DataFrame:
        df = self._labels.copy()
        # Transform gender
        df["gender"] = _category_to_float(df["gender"], {"female": 0, "male": 1})

        # Fix bools and float32s
        for attr in df.columns:
            df_attr = df[attr]
            dtype = df_attr.dtypes
            if dtype == bool:
                df[attr] = _bool_to_binary(df_attr)
            elif dtype == np.float32:
                # Poses
                if "headPose" in attr:
                    df[attr] = _min_max_scaler(df_attr, min=-90, max=90)

                # Age
                elif df_attr.min() >= 0 and df_attr.max() > 1:
                    df[attr] = _min_max_scaler(df_attr)

                # Already between 0 and 1.
                else:
                    continue

        # Remove redundant columns
        df = df.select_dtypes(include=["float32"])
        return df

    def init_files(self) -> tuple[FileJar, pd.DataFrame]:
        """
        Initilizes the label files related to the dataset.
        Extract labels from the dataset and saves them with a
        FileJar as well as a pd.DataFrame.

        Returns:
            tuple[FileJar, pd.DataFrame]: FileJar with save paths and
                a pd.DataFrame containing labels.
        """
        file_jar = FileJar(self.get_path())
        df = file_jar.get_file(self.DS_LABELS_NAME, pd.read_pickle)
        if df is None:
            res = self._parse_json()
            self._img_paths = self._calc_img_paths()
            return res
        else:
            print(f"======= Labels already extracted, skipping extraction =======")
            self._img_paths = self._calc_img_paths()
            return file_jar, df

    def _calc_img_paths(self) -> list[Path]:
        ext = next(Path(self.get_image_dir()).iterdir()).name.split(".")[-1]
        return list(
            [
                Path(f"{self.get_image_dir() / s}.{ext}")
                for s in zero_padded_iterator(0, 70000, 5)
            ]
        )

    def _parse_json(self) -> tuple[FileJar, pd.DataFrame]:
        """
        Extract labels from the dataset and saves them with a
        FileJar as well as a pd.DataFrame. Extraction is done by
        multiprocessing the tasks, utilizing `multiprocessing.cpu_count()`
        number of workers.

        Raises:
            FileNotFoundError: If directory does not exist for the
                labels.

        Returns:
            tuple[FileJar, pd.DataFrame]: FileJar with save paths and
                a pd.DataFrame containing labels.
        """
        # Loop through all json files, save all filenames
        label_dir_path = self.get_path() / self.DS_DIR_LABELS
        if label_dir_path.is_dir():
            json_files = [
                pos_json
                for pos_json in os.listdir(label_dir_path)
                if pos_json.endswith(".json")
            ]
        else:
            raise FileNotFoundError(f"Directory {label_dir_path} not found!")

        # Create iterable object for multprocesssing work
        iter = [(index, json_f) for index, json_f in enumerate(json_files)]

        # Setup multiprocess
        nb_workers = cpu_count()
        print(f"======= Parsing JSON using {nb_workers} workers =======")
        pool = Pool(processes=nb_workers)

        # Parse all json files using multiprocessing
        res = pool.imap_unordered(self._mp_json_parser, iter)

        # Save res
        label_frames = [
            df for df in tqdm(res, total=len(json_files)) if type(df) != list
        ]
        df_conc = pd.concat(label_frames)

        # Convert to correct types
        df_conc[df_conc.select_dtypes(np.float64).columns] = df_conc.select_dtypes(
            np.float64
        ).astype(np.float32)
        df_conc[df_conc.select_dtypes(object).columns] = df_conc.select_dtypes(
            object
        ).astype("category")

        # Create FileJar and store the file.
        file_jar = FileJar(self.get_path())
        file_jar.store_file(self.DS_LABELS_NAME, df_conc.to_pickle)

        print(f"======= Done parsing JSON =======")

        return file_jar, df_conc

    # Multiprocess function
    def _mp_json_parser(self, args: tuple[int, str]) -> pd.DataFrame:
        """
        Multiprocessing function used to prase one JSON-file into
        a pd.DataFrame row.

        Args:
            args (tuple[int, str]): Image index, JSON-filename

        Returns:
            pd.DataFrame: The extracted labels of the JSON-file.
        """
        index = args[0]
        json_f = args[1]

        # Parse json
        with open(
            os.path.join(self.get_path() / self.DS_DIR_LABELS, json_f)
        ) as json_file:
            json_obj = json.load(json_file)
            if not json_obj:
                return []

            # Load json
            json_dict = json_obj[0]["faceAttributes"]
            json_norm = pd.json_normalize(json_dict)

            # Fix nestled hairColor column
            nestled_colname = [
                "hair.hairColor.brown",
                "hair.hairColor.blond",
                "hair.hairColor.black",
                "hair.hairColor.red",
                "hair.hairColor.gray",
                "hair.hairColor.other",
            ]

            if not json_norm["hair.hairColor"][0]:
                for col in nestled_colname:
                    json_norm[col] = np.NaN
                json_norm["hair.hairColor.mp"] = np.NaN
                json_norm.drop(["hair.hairColor"], axis=1, inplace=True)
            else:
                # Get colnames
                prefix = "hair.hairColor."
                wrong_order_colnames = [
                    prefix + d["color"] for d in json_norm["hair.hairColor"].tolist()[0]
                ]

                # Normalize json
                json_hair = pd.json_normalize(json_norm["hair.hairColor"]).set_axis(
                    wrong_order_colnames, axis=1
                )

                # Rearrange columns to standardized order
                json_hair = json_hair[nestled_colname]

                # Parse the confidence of the hair color
                json_hair_parsed = json_hair.apply(
                    lambda x: x[0]["confidence"], result_type="broadcast"
                )
                json_hair_parsed = json_hair_parsed.astype("float64")

                # Add most probable hairColor column
                json_hair_parsed["hair.hairColor.mp"] = json_hair_parsed.astype(
                    float
                ).idxmax(axis=1)
                json_norm = pd.concat(
                    [json_norm.drop(["hair.hairColor"], axis=1), json_hair_parsed],
                    axis=1,
                )

            # Fix nestled accessories column
            nestled_colname = [
                "accessories.glasses",
                "accessories.headwear",
                "accessories.mask",
            ]
            if not json_norm["accessories"][0]:
                for col in nestled_colname:
                    json_norm[col] = 0.0
                json_norm.drop(["accessories"], axis=1, inplace=True)
            else:
                # Get colnames
                prefix = "accessories."
                wrong_order_colnames = [
                    prefix + d["type"] for d in json_norm["accessories"].tolist()[0]
                ]

                # Normalize json
                json_acc = pd.json_normalize(json_norm["accessories"]).set_axis(
                    wrong_order_colnames, axis=1
                )

                # Parse the confidence of the accessories
                json_acc_parsed = json_acc.apply(
                    lambda x: x[0]["confidence"], result_type="broadcast"
                )
                json_acc_parsed = json_acc_parsed.astype("float64")

                # Looking for missing columns
                wrong_order_colnames_missing = [
                    col for col in nestled_colname if col not in wrong_order_colnames
                ]

                # Add missing columns and values
                for col in wrong_order_colnames_missing:
                    json_acc_parsed[col] = 0.0

                # Rearrange columns to standardized order
                json_acc_parsed = json_acc_parsed[nestled_colname]

                # Add the parsed accessories columns
                json_norm = pd.concat(
                    [json_norm.drop(["accessories"], axis=1), json_acc_parsed], axis=1
                )

            json_norm.index = [index]
        return json_norm


def _category_to_float(df, d: dict):
    return df.replace(d).astype(int).astype(np.float32)


def _bool_to_binary(df):
    return df.astype(int).astype(np.float32)


def _min_max_scaler(df, min=None, max=None):
    """
    Min-max scaler scaling dataframe from min to max, such that
    all values lies within this range. Calculates min and max if not provided.
    """
    min_vals = df.min()
    if not min is None:
        min_vals = df.copy()
        min_vals.values[:] = min

    max_vals = df.max()
    if not min is None:
        max_vals = df.copy()
        max_vals.values[:] = max

    return (df - min_vals) / (max_vals - min_vals)
