from typing import Literal
import pandas as pd
import logging

import os
from pathlib import Path

Dataset = Literal["CSC", "MP", "Paraphrase", "VariErrNLI"]


def load_dataset(dataset: Dataset) -> pd.DataFrame:
    root = (
        Path(os.environ["DSS_HOME"]) / "lewidi-data" / "data_practice_phase" / dataset
    )
    ds = root / f"{dataset}_train.json"
    assert ds.exists(), ds.absolute()

    df = pd.read_json(ds, orient="index")
    df.reset_index(inplace=True)
    return df


def load_template(dataset: Dataset, template_id: str) -> str:
    root = Path(__file__).parent / "prompt_templates"
    template = root / f"{dataset}_{template_id}.txt"
    assert template.exists(), template.absolute()
    with open(template, "r") as f:
        return f.read()


def enable_logging():
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
