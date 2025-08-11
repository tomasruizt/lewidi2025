from lewidi_lib import (
    assign_cols_perf_metrics,
    enable_logging,
    join_dataset_and_preds,
    load_dataset,
    load_listof_parquets,
    preds_file,
    process_rdf,
)


def main():
    dataset = "CSC"
    split = "train"
    task = "soft-label"
    file = preds_file(
        dataset=dataset,
        split=split,
        model_id="Qwen/Qwen3-32B",
        template="60",
        run_name="1000ex_10loops",
    )
    rdf = load_listof_parquets([file])
    rdf = process_rdf(
        rdf, discard_invalid_pred=True, task=task, response_contains_steps=True
    )
    ddf = load_dataset(dataset=dataset, split=split, task=task)
    joint_df = join_dataset_and_preds(ddf, rdf)
    joint_df = assign_cols_perf_metrics(joint_df, task=task)
    return joint_df


if __name__ == "__main__":
    enable_logging()
    joint_df = main()
    print(joint_df)
