from lewidi_lib import create_zip_file, enable_logging
from lewidi_regression import dump_submissions_regression

enable_logging()


datasets = ["CSC"]
files = dump_submissions_regression(datasets)
zip_file = create_zip_file(files)
print(f"\nAll submission files have been zipped into {zip_file}")
