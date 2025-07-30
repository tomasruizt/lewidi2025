from inference_lib import Args, run_inference
from lewidi_lib import enable_logging

enable_logging()
args = Args()
run_inference(args)
