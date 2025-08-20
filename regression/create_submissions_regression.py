from lewidi_lib import create_zip_file, enable_logging, submissions_root_regression
from lewidi_regression import dump_submissions_regression

enable_logging()


datasets = ["CSC", "Paraphrase", "MP"]
tgt_dir = submissions_root_regression()
files = dump_submissions_regression(datasets, tgt_dir=tgt_dir)
zip_file = create_zip_file(files, tgt_dir=tgt_dir)
print(f"\nAll submission files have been zipped into {zip_file}")
