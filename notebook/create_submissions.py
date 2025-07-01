from lewidi_lib import (
    compute_average_baseline,
    dump_submission_file,
    enable_logging,
    load_dataset,
    load_preds_for_submission,
    assert_submission_nrows_as_expected,
    reorder_like_ddf,
    submissions_root,
)
import zipfile
import os

enable_logging()

datasets = ["VariErrNLI", "CSC", "MP", "Paraphrase"]
split = "test_clear"
files = []
for dataset in datasets:
    rdf = load_preds_for_submission(dataset, split)
    ddf = load_dataset(dataset=dataset, split=split, parse_tgt=False)
    assert_submission_nrows_as_expected(rdf, ddf)
    model_avg = compute_average_baseline(rdf)
    model_avg = reorder_like_ddf(rdf=model_avg, ddf=ddf)
    tgt_file = dump_submission_file(rdf=model_avg, dataset=dataset)
    files.append(tgt_file)


zip_file = submissions_root() / "res.zip"
with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file_path in files:
        zipf.write(file_path, os.path.basename(file_path))

print(f"\nAll submission files have been zipped into {zip_file}")
