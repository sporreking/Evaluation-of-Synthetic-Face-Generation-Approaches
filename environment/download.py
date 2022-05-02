from enum import Enum
import git
from git import RemoteProgress
from pathlib import Path
from tqdm import tqdm
import gdown

# * Used for indicating how to download different pre-trained models
StorageType = Enum("StorageType", "MESSAGE GOOGLE_DRIVE")

# * Add repositories that should be downloaded to this dictionary
REPOSITORIES_TO_DOWNLOAD = {
    "unetgan": (
        "https://github.com/boschresearch/unetgan.git",
        "master",
        (
            StorageType.MESSAGE,
            "Make sure to download pretrained models! Check 'unetgan/pretrained_model/README.md'",
            None,
        ),
    ),
    "stylegan2ada": (  # Official pytorch version
        "https://github.com/NVlabs/stylegan2-ada-pytorch.git",
        "main",
        None,
    ),
    "styleswin": (
        "https://github.com/microsoft/StyleSwin.git",
        "main",
        (
            StorageType.GOOGLE_DRIVE,
            "https://drive.google.com/uc?id=1OjYZ1zEWGNdiv0RFKv7KhXRmYko72LjO",
            "FFHQ_256.pt",
        ),
    ),
}

# Where to download repositories to
ROOT = Path("environment")


class CloneProgress(RemoteProgress):
    """Progress logger for Git cloning."""

    def __init__(self, repo: str):
        super().__init__()
        self._pbar = tqdm(desc=repo)

    def update(self, op_code, cur_count, max_count=None, message=""):
        self._pbar.total = max_count
        self._pbar.n = cur_count
        self._pbar.refresh()


def main():
    """Downloads all repositories."""

    tqdm.write("Downloading repositories...\n")
    for repo, (url, branch, pt_info) in REPOSITORIES_TO_DOWNLOAD.items():
        output_path = ROOT / repo
        if output_path.is_file():
            raise RuntimeError(f"Download location occupied by file: '{output_path}'")
        elif output_path.is_dir():
            tqdm.write(f"Repository '{repo}' already downloaded, skipping...\n")
        else:
            git.Repo.clone_from(
                url, ROOT / repo, progress=CloneProgress(repo), branch=branch
            )
        if pt_info is not None:
            _handle_pt_info(repo, *pt_info)
            print()
    tqdm.write("Done!")


def _handle_pt_info(repo_name: str, type: StorageType, info: str, output_file: str):
    if type == StorageType.MESSAGE:
        tqdm.write(f"IMPORTANT --[{repo_name}]-> {info}\n")
    else:
        out_file = ROOT / repo_name / "pretrain" / output_file
        if out_file.is_dir():
            raise RuntimeError(f"Download target occupied by directory: '{out_file}'")
        elif out_file.is_file():
            tqdm.write(
                f"Pretrained model for '{repo_name}' already downloaded, skipping...\n"
            )
            return
        elif not out_file.parent.is_dir():
            out_file.parent.mkdir(exist_ok=True)

        # Determine how to download
        if type == StorageType.GOOGLE_DRIVE:
            gdown.download(info, str(out_file), quiet=False)
        else:
            raise ValueError(f"Invalid storage type: '{type.name}'")


if __name__ == "__main__":
    main()
