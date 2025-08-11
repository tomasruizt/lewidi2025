from regress_lm import core
from regress_lm import rlm

# Create RegressLM with max input token length.
reg_lm = rlm.RegressLM.from_default(max_input_len=2048)

# Example (x,y) pairs, which can be fine-tuned against.
examples = [core.Example(x="hello", y=0.3), core.Example(x="world", y=-0.3)]
reg_lm.fine_tune(examples)

# Query inputs.
query1, query2 = core.ExampleInput(x="hi"), core.ExampleInput(x="bye")
samples1, samples2 = reg_lm.sample([query1, query2], num_samples=128)
