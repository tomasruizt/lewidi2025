from logging import getLogger
from lewidi_lib import configure_pandas_display, enable_logging
from lewidi_regression import FullEval, run_all_evals
import pandas as pd


logger = getLogger(__name__)

if __name__ == "__main__":
    enable_logging()
    configure_pandas_display()

    dataset = "Paraphrase"
    files = [
        f"/mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-2b-2b-prefixlm/{dataset}/preds.parquet",
        f"/mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-2b-2b-prefixlm/CSC,MP,Paraphrase/preds.parquet",
    ]

    full_evals = []
    for idx, file in enumerate(files):
        logger.info(f"Evaluating {file}")
        eval_df = pd.read_parquet(file)
        full_eval = run_all_evals(eval_df).assign_col("idx", idx)
        full_evals.append(full_eval)
    full_evals: FullEval = sum(full_evals).query(f"dataset == '{dataset}'")
    full_evals.log_self()
