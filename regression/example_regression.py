from regress_lm.models.pytorch import t5gemma_model
from regress_lm.models.pytorch.model import PyTorchFineTuner
from regress_lm import core, rlm

device = "cuda:0"

print("Starting model...")
t5 = t5gemma_model.T5GemmaModel(
    "google/t5gemma-s-s-prefixlm",
    model_kwargs={"attn_implementation": "eager"},
)
t5.to(device)
reg_lm = rlm.RegressLM(t5, PyTorchFineTuner(t5))
print("Done")

examples = [
    core.Example(x="hello", y=0.3),
    core.Example(x="world", y=-0.3),
]
print("Fine-tuning...")
reg_lm.fine_tune(examples)
print("Done")


print("Sampling...")
query1, query2 = core.ExampleInput(x="hi"), core.ExampleInput(x="bye")
# samples1, samples2 = reg_lm.sample([query1, query2], num_samples=128)
examples = reg_lm.model.convert_inputs([query1, query2])
examples = {k: v.to(device) for k, v in examples.items()}

_, output_floats = reg_lm.model.decode(examples, num_samples=10)
print(output_floats)
print("Done")
