from lewidi_lib import configure_pandas_display, enable_logging
from lewidi_regression import explode_preds_and_discard_invalid, run_all_evals
import pandas as pd


if __name__ == "__main__":
    enable_logging()
    configure_pandas_display()

    #file = "/mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-s-s-prefixlm/MP/preds.parquet"
    # file = "/mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-s-s-prefixlm/Paraphrase/preds.parquet"
    file = "/mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-s-s-prefixlm/CSC/preds.parquet"
    # file = "/mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-2b-2b-prefixlm/CSC/preds.parquet"
    eval_df = pd.read_parquet(file)
    eval_df = explode_preds_and_discard_invalid(eval_df)
    run_all_evals(eval_df)
