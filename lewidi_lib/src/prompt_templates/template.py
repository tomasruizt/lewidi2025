from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping
import nltk

from lewidi_lib import Dataset

from . import templates_root


class Template(ABC):
    @abstractmethod
    def make_prompt(self, data: Mapping) -> str:
        """Fill the template with data from the given row mapping."""
        pass


def load_template(dataset: Dataset, template_id: int) -> str:
    template = templates_root / f"{dataset}_{str(template_id)}.txt"
    return load_template_file(template)


@dataclass
class PredTemplate(Template):
    dataset: Dataset
    template_id: str

    def __post_init__(self):
        self.template: str = load_template(self.dataset, self.template_id)

    def make_prompt(self, data: Mapping) -> str:
        return self.template.format(text=data["text"])


def load_template_file(file: str | Path) -> str:
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"Template file '{file.absolute()}' not found")
    return file.read_text()


@dataclass
class JudgeCoTSentencesTemplate(Template):
    pred_template: PredTemplate
    judge_template_file = "reasoning_trace_eval2.txt"

    def __post_init__(self):
        self.judge_template = load_template_file(
            templates_root / self.judge_template_file
        )
        nltk.download("punkt_tab")

    def make_prompt(self, data: Mapping) -> str:
        llm_problem = self.pred_template.make_prompt(data)
        steps: list[str] = [{"text": s} for s in nltk.sent_tokenize(data["reasoning"])]
        prompt = self.judge_template.format(
            PROBLEM=llm_problem, STEPS=json.dumps(steps, indent=2)
        )
        return prompt


def in_qwen3_format(reasoning: str, output: str) -> str:
    return f"""
<think>
{reasoning}
</think>

{output}
"""


@dataclass
class JudgeOutcomeTemplate(Template):
    pred_template: PredTemplate
    judge_template_file = "judge_eval.txt"

    def __post_init__(self):
        self.judge_template = load_template_file(
            templates_root / self.judge_template_file
        )

    def make_prompt(self, data: Mapping) -> str:
        llm_problem = self.pred_template.make_prompt(data)
        llm_solution = in_qwen3_format(data["reasoning"], data["response"])
        prompt = self.judge_template.format(
            llm_problem=llm_problem,
            llm_solution=llm_solution,
        )
        return prompt


@dataclass
class ReformatTemplate(Template):
    pred_template: PredTemplate
    reformat_template_file = "reformat.txt"

    def __post_init__(self):
        self.reformat_template = load_template_file(
            templates_root / self.reformat_template_file
        )

    def make_prompt(self, data: Mapping) -> str:
        llm_problem = self.pred_template.make_prompt(data)
        llm_response = data["reasoning"]
        prompt = self.reformat_template.format(
            problem=llm_problem,
            response=llm_response,
        )
        return prompt
