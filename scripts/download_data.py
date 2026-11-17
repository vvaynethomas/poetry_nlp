"""Fetch the Complete Poetry Foundation dataset from Kaggle into data/raw/.

Uses the official Kaggle CLI, which needs an API token in ~/.kaggle/kaggle.json
(Kaggle -> Settings -> API -> Create New Token) and acceptance of the dataset's
terms on its Kaggle page. Without the CLI, download the zip by hand from
https://www.kaggle.com/datasets/johnhallman/complete-poetryfoundationorg-dataset
and unzip it so that data/raw/kaggle_poem_dataset.csv exists.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poetry_nlp import config  # noqa: E402


def main():
    target = config.RAW_CSV
    if target.exists():
        print(f"{target.relative_to(ROOT)} already present.")
        return
    if shutil.which("kaggle") is None:
        sys.exit("The Kaggle CLI is not installed (`pip install kaggle`).\n" + __doc__)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", config.KAGGLE_DATASET,
                    "-p", str(target.parent), "--unzip"], check=True)
    if not target.exists():
        found = sorted(target.parent.glob("*.csv"))
        sys.exit(f"Download finished but {target.name} was not found; got {found}.")
    print(f"Saved {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
