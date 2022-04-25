import git
from git import RemoteProgress
from pathlib import Path
from tqdm import tqdm

# * Add repositories that should be downloaded to this dictionary
REPOSITORIES_TO_DOWNLOAD = {
    "unetgan": (
        "https://github.com/boschresearch/unetgan.git",
        "master",
        "Make sure to download pretrained models! Check 'unetgan/pretrained_model/README.md'",
    ),
    "stylegan2ada": (  # Official pytorch version
        "https://github.com/NVlabs/stylegan2-ada-pytorch.git",
        "main",
        None,
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
    for repo, (url, branch, info) in REPOSITORIES_TO_DOWNLOAD.items():
        output_path = ROOT / repo
        if output_path.is_file():
            raise RuntimeError(f"Download location occupied by file: '{output_path}'")
        elif output_path.is_dir():
            tqdm.write(f"Repository '{repo}' already downloaded, skipping...")
        else:
            git.Repo.clone_from(
                url, ROOT / repo, progress=CloneProgress(repo), branch=branch
            )
        if info is not None:
            tqdm.write(f"\nIMPORTANT --[{repo}]-> {info}\n")
    tqdm.write("Done!")


if __name__ == "__main__":
    main()
