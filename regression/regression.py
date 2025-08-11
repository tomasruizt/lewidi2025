import json
from regress_lm.models.pytorch.model import PyTorchFineTuner
from regress_lm.models.pytorch import t5gemma_model
from lewidi_lib import enable_logging, load_dataset
from pathlib import Path
from regress_lm import core, rlm
import pandas as pd
import numpy as np
import torch

device = "cuda:0"


def load_data() -> pd.DataFrame:
    dataset = "MP"
    split = "train"
    task = "perspectivist"
    return load_dataset(dataset=dataset, split=split, task=task)


def make_examples(ddf: pd.DataFrame) -> list[core.Example]:
    template = (Path(__file__).parent / "MP_template.txt").read_text()
    examples = []
    for _, row in ddf.iterrows():
        post = row["text"]["post"]
        reply = row["text"]["reply"]
        personas = row["annotator_metadata"]
        tgts = row["target"]
        assert len(tgts) == len(personas)
        for persona, tgt in zip(personas, tgts):
            persona_str = json.dumps(persona, indent=2)
            prompt = template.format(post=post, reply=reply, persona=persona_str)
            examples.append(core.Example(x=prompt, y=float(tgt)))
    return examples


def create_model() -> rlm.RegressLM:
    t5 = t5gemma_model.T5GemmaModel(
        "google/t5gemma-s-s-prefixlm",
        max_input_len=512,  # Reduced from 2048 to save memory
        model_kwargs={
            "attn_implementation": "eager",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        },
    )
    t5.to(device)
    t5.compile()

    # Use LoRA fine-tuner for memory efficiency
    #lora_finetuner = PEFTLoRAFineTuner(model=t5, lora_rank=1)

    model = rlm.RegressLM(t5, PyTorchFineTuner(t5))
    return model


def inference(
    model: rlm.RegressLM, exs: list[core.ExampleInput], num_samples: int = 10
) -> np.ndarray:
    examples = model.model.convert_inputs(exs)
    examples = {k: v.to(device) for k, v in examples.items()}
    _, output_floats = model.model.decode(examples, num_samples=num_samples)
    return output_floats


if __name__ == "__main__":
    enable_logging()
    ddf = load_data()
    model = create_model()

    # train_exs = make_examples(ddf.sample(n=10))
    # print(len(train_exs))
    # model.fine_tune(train_exs, batch_size=1)

    test_exs = make_examples(ddf.sample(n=2))
    preds = inference(model, test_exs, num_samples=10)
    print(preds)
