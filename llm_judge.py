import logging
from judge_lib import (
    JudgeArgs,
    create_judge_batch,
    make_judge_model,
    process_batch,
)
from lewidi_lib import enable_logging


if __name__ == "__main__":
    enable_logging()
    logger = logging.getLogger(__name__)

    args = JudgeArgs()
    logger.info("Args: %s", args.model_dump_json())
    batch = create_judge_batch(args)
    if len(batch) == 0:
        logger.info("No examples to judge")
        exit(0)

    model = make_judge_model(args)
    if args.dry_run:
        logger.info("Dry run. Not running the judge.")
    else:
        process_batch(model, args, batch)
