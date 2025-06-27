from logging import getLogger
from pathlib import Path

from lewidi_lib import enable_logging
import pandas as pd
from pydantic_settings import BaseSettings

logger = getLogger(__name__)


class Args(BaseSettings, cli_parse_args=True):
    """Command line arguments for the metrics plotting script."""

    file: Path


def extract_reasoning(response: dict) -> str | None:
    reasonings = [
        p["text"]
        for p in response["candidates"][0]["content"]["parts"]
        if is_thought(p)
    ]
    if len(reasonings) > 1:
        logger.warning("Found %d reasoning parts. Expected 1.", len(reasonings))
    if len(reasonings) > 0:
        return reasonings[0]
    return None


def extract_text(response: dict) -> str | None:
    texts = [
        p["text"]
        for p in response["candidates"][0]["content"]["parts"]
        if not is_thought(p)
    ]
    if len(texts) > 1:
        logger.warning("Found %d text parts. Expected 1.", len(texts))
    if len(texts) > 0:
        return texts[0]
    return None


def is_thought(part: dict) -> bool:
    return "thought" in part and part["thought"]


def extract_finish_reason(response: dict) -> str:
    return response["candidates"][0]["finishReason"]


def main(args=None):
    enable_logging()

    if args is None:
        args = Args()
    folder = args.file.parent
    df = pd.read_json(args.file, lines=True)
    new_df = df.assign(
        reasoning=df["response"].apply(extract_reasoning),
        response=df["response"].apply(extract_text),
        success=df["response"].apply(extract_finish_reason) == "STOP",
    )
    new_df = new_df.drop(columns=drop_cols)

    new_df_jsonl = folder / "responses.jsonl"
    new_df.to_json(new_df_jsonl, orient="records", lines=True)
    logger.info("Wrote file: %s", new_df_jsonl.absolute())


drop_cols = [
    "status",
    "processed_time",
    "request",
    # "response",  # we need this col
]

if __name__ == "__main__":
    args = Args(
        file="/home/tomasruiz/datasets/dss_home/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t31/CSC/allexs_20loops/judge/gemini-2.5-flash/t2/lewidi-judge-3ex-3loops/predictions.jsonl"
    )
    main(args)
