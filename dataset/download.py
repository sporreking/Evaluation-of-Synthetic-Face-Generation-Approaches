import git
from git import RemoteProgress
from pathlib import Path
from tqdm import tqdm

# * Add repositories that should be downloaded to this dictionary
# format for each entry should be:
# name : (url to repo, branch, msg)
# Use `None` for irrelevant entries.
REPOSITORIES_TO_DOWNLOAD = {
    "FFHQ": {
        "LABELS": (  # Setup FFHQ labels
            "https://github.com/DCGM/ffhq-features-dataset.git",
            "master",
            None,
        ),
        "IMAGES": {
            256: (  # Setup FFHQ 256 images
                None,
                None,
                "FFHQ 256 dataset instructions:\n"
                "1. Download dataset from (must register account on Kaggle):\n"
                + "https://www.kaggle.com/datasets/xhlulu/flickrfaceshq-dataset-nvidia-resized-256px?resource=download\n"
                + "2. Unzip and place image files in directory:\n"
                + "dataset/FFHQ_256/image",
            ),
        },
    }
}

# Where to download repositories to
ROOT = Path("dataset")


def main():
    """Downloads all repositories."""
    tqdm.write("Downloading repositories...\n")
    for name, label_image_dict in REPOSITORIES_TO_DOWNLOAD.items():
        # Images
        for res, (url, branch, msg) in label_image_dict["IMAGES"].items():
            _setup_repos(url, _get_dataset_name(name, res) + "/image", branch, msg)

        # Labels
        if "LABELS" in label_image_dict:
            url, branch, msg = label_image_dict["LABELS"]
            _setup_repos(url, _get_dataset_labels_name(name), branch, msg)


def _setup_repos(url, repo, branch, msg):
    if url is None:
        _setup_directory(ROOT / repo)
    else:
        output_path = ROOT / repo
        if output_path.is_file():
            raise RuntimeError(f"Download location occupied by file: '{output_path}'")
        elif output_path.is_dir():
            tqdm.write(f"Repository '{repo}' already downloaded, skipping...\n")
        else:
            git.Repo.clone_from(
                url, ROOT / repo, progress=CloneProgress(repo), branch=branch
            )
    if msg is not None:
        tqdm.write("-" * len(msg))
        tqdm.write(msg)
        tqdm.write("-" * len(msg))
    tqdm.write("Done!")


def _get_dataset_labels_name(name: str) -> str:
    # Assumes sames labels for different resolutions
    return "_".join([name, "LABELS"])


def _get_dataset_name(name: str, res: int) -> str:
    return "_".join([name, str(res)])


def _setup_directory(dir_path: Path) -> None:
    if dir_path.is_file():
        RuntimeError(f"File found at {dir_path}, file should not exist.")
    else:
        dir_path.mkdir(parents=True, exist_ok=True)


class CloneProgress(RemoteProgress):
    """Progress logger for Git cloning."""

    def __init__(self, repo: str):
        super().__init__()
        self._pbar = tqdm(desc=repo)

    def update(self, op_code, cur_count, max_count=None, message=""):
        self._pbar.total = max_count
        self._pbar.n = cur_count
        self._pbar.refresh()


if __name__ == "__main__":
    main()
