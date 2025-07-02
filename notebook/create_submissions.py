from lewidi_lib import create_zip_file, dump_submission_files, enable_logging

enable_logging()

datasets = ["VariErrNLI", "CSC", "MP", "Paraphrase"]
files = dump_submission_files(datasets)
zip_file = create_zip_file(files)
print(f"\nAll submission files have been zipped into {zip_file}")
