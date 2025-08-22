from logging import getLogger
from lewidi_lib import configure_pandas_display, enable_logging
from lewidi_regression import explode_preds_and_discard_invalid, run_all_evals
import pandas as pd


logger = getLogger(__name__)

if __name__ == "__main__":
    enable_logging()
    configure_pandas_display()

    dataset = "CSC"
    files = [
        f"/mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-2b-2b-prefixlm/{dataset}/preds.parquet",
        f"/mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-s-s-prefixlm/{dataset}/preds.parquet",
    ]
    for file in files:
        logger.info(f"Evaluating {file}")
        eval_df = pd.read_parquet(file)
        # eval_df = explode_preds_and_discard_invalid(eval_df)
        run_all_evals(eval_df)
