from regress_lm import core
from regress_lm import rlm

import numpy as np

device = "cuda:0"

print("Starting model...")
reg_lm = rlm.RegressLM.from_default(max_input_len=2048)
reg_lm.model.to(device)
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

_, output_floats = reg_lm.model.decode(examples, 128)
print(output_floats)
print("Done")
