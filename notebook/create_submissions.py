from lewidi_lib import (
    create_zip_file,
    dump_submission_files_perspectivist,
    dump_submission_files_softlabel,
    enable_logging,
)

enable_logging()

datasets = ["VariErrNLI", "Paraphrase"]  # "CSC", "MP"]
files_pe = dump_submission_files_perspectivist(datasets)
files_soft = dump_submission_files_softlabel(datasets)
zip_file = create_zip_file(files_pe + files_soft)
print(f"\nAll submission files have been zipped into {zip_file}")
