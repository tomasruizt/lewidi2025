# Instructions for Reasoning Trace Evaluation

You are evaluating the quality of a reasoning trace that attempts to justify a proposed distribution over interpretations (e.g., sarcasm level) for a given context-response pair. Your goal is to assess how well the trace models the pragmatic thinking of a typical human annotator and how convincingly it argues for its proposed distribution.

Please output a single floating-point score between 0.0 (very poor reasoning) and 1.0 (excellent reasoning), based on the following criteria:

## Evaluation Criteria

1. Pragmatic Grounding and Use of Heuristics

- Does the trace rely on realistic, common-sense social and linguistic reasoning?
- Does it use plausible heuristics that a typical human would apply in context (e.g., tone markers, potential danger, typical intentions behind common phrases)?
- Does it avoid abstract, speculative, or overly academic reasoning disconnected from the situation?

2. Decisive Inference, Not Just Listing

- Does the trace go beyond merely listing possibilities and argue for one interpretation as more likely than the others?
- Does it appropriately weigh stronger and weaker interpretations, rather than treating them all equally?
  
3. Justification for the Distribution’s Shape

- Does the trace clearly explain why its proposed distribution is concentrated or spread out?
- Is there a logical and well-supported connection between the reasoning and the shape of the predicted distribution?

## Scoring Guidelines

- 0.75 to 1.00: Trace is excellent across all three criteria — pragmatic, decisive, and tightly aligned with the proposed distribution.
- 0.59 to 0.74: Solid trace with minor flaws — maybe slightly under-argued or slightly abstract at times, but mostly strong.
- 0.25 to 0.49: Mixed — has some valuable insights, but suffers from indecision, poor weighting, or reliance on weak heuristics.
- 0.00 to 0.24: Weak — overly speculative, fails to make a convincing case, or does not justify the distribution’s shape.

Please consider the full trace, not just isolated statements, and use your best judgment in applying these guidelines.
Be very strict with the criteria! Ideally, the scores should be equally distributed across the bins.
Output only the score (e.g., 0.81) and nothing else.

# Begin

## Context-Response Pair

{example}

## Reasoning Trace

{reasoning}
