from inference_lib import Args, create_all_batches
from llmlib.vllm_model import ModelvLLM
from llmlib.base_llm import LlmReq
from llmlib.gemini.gemini_code import GeminiAPI
from llmlib.mock_model import MockModel
import logging
from tqdm import tqdm
from lewidi_lib import (
    convert_output_to_parquet,
    enable_logging,
    dump_response,
    postprocess_response,
    using_vllm_server,
)

logger = logging.getLogger(__name__)


def run_many_inferences(args: Args) -> None:
    batches: list[LlmReq] = create_all_batches(args)
    logger.info("Total num of examples: %d", len(batches))

    pbar = tqdm(total=len(batches))

    model = make_model(args)
    responses = model.complete_batchof_reqs(batch=batches)
    for response in responses:
        response = postprocess_response(response)
        dump_response(tgt_file=args.tgt_file, response=response)
        pbar.update(1)


def make_model(args: Args):
    if args.model_id == "test":
        return MockModel()

    if args.model_id.startswith("gemini"):
        model = GeminiAPI(
            model_id=args.model_id,
            max_n_batching_threads=args.remote_call_concurrency,
            include_thoughts=True,
            location="global",
        )
        return model

    model = ModelvLLM(
        remote_call_concurrency=args.remote_call_concurrency,
        model_id=args.model_id,
        port=args.vllm.port,
        timeout_secs=args.timeout_secs,
    )
    return model


if __name__ == "__main__":
    enable_logging()
    args = Args()
    logger.info("Args: %s", args.model_dump_json())
    with using_vllm_server(args.model_id, args.vllm):
        run_many_inferences(args)
    convert_output_to_parquet(args.tgt_file)
