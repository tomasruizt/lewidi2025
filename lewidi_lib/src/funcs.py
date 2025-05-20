import pandas as pd
import logging

import os
from pathlib import Path


def load_dataset(dataset_name: str) -> pd.DataFrame:
    root = (
        Path(os.environ["DSS_HOME"])
        / "lewidi-data"
        / "data_practice_phase"
        / dataset_name
    )
    ds = root / f"{dataset_name}_train.json"
    assert ds.exists(), ds.absolute()

    possible_answers = {"CSC": list(range(1, 7))}

    df = pd.read_json(ds, orient="index")
    df.reset_index(inplace=True)
    return df


def enable_logging():
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
