from src.dataset.Dataset import Dataset
from src.util.FileJar import FileJar

import os, json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np


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

    # Dataset name
    DS_NAME = "FFHQ"

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
        name = self.DS_NAME + "_" + str(resolution)
        super(FFHQDataset, self).__init__(name)
        self._resolution = resolution

    def get_resolution(self) -> int:
        """
        Returns the resolution of the dataset.

        Returns:
            int: The resolution of the dataset.
        """
        return self._resolution

    def init_files(self) -> tuple[FileJar, pd.DataFrame]:
        """
        Initilizes the label files related to the dataset.
        Extract labels from the dataset and saves them with a
        FileJar as well as a pd.DataFrame.

        Returns:
            tuple[FileJar, pd.DataFrame]: FileJar with save paths and
                a pd.DataFrame containing labels.
        """
        file_jar = FileJar(self._ds_dir)
        df = file_jar.get_file(self.DS_LABELS_NAME, pd.read_pickle)
        if df is None:
            return self._parse_json()
        else:
            print(f"======= Labels already extracted, skipping extraction =======")
            return file_jar, df

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
        label_dir_path = self._ds_dir / self.DS_DIR_LABELS
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
        file_jar = FileJar(self._ds_dir)
        file_jar.store_file(self.DS_LABELS_NAME, df_conc.to_pickle)

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
        with open(os.path.join(self._ds_dir / self.DS_DIR_LABELS, json_f)) as json_file:
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
