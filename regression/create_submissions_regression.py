from lewidi_lib import create_zip_file, enable_logging, submissions_root_regression
from lewidi_regression import dump_submissions_regression

enable_logging()


datasets = ["CSC", "Paraphrase"]
files = dump_submissions_regression(datasets)
zip_file = create_zip_file(files, root=submissions_root_regression())
print(f"\nAll submission files have been zipped into {zip_file}")
