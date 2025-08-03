from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from logging import getLogger
from pathlib import Path
from typing import Iterable, Mapping
import json_repair
import nltk

from lewidi_lib import Dataset

from . import templates_root

logger = getLogger(__name__)


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


@dataclass
class PredTemplateWithAnnotatorMetadata(PredTemplate):
    def make_prompt(self, data: Mapping) -> str:
        metadata = json.dumps(data["annotator_metadata"], indent=2)
        return self.template.format(text=data["text"], annotator_metadata=metadata)


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


@dataclass
class JudgeCoTParagraphsTemplate(Template):
    pred_template: PredTemplate
    judge_template_file = "reasoning_trace_eval22.txt"

    def __post_init__(self):
        self.judge_template = load_template_file(
            templates_root / self.judge_template_file
        )

    def make_prompt(self, data: Mapping) -> str:
        llm_problem = self.pred_template.make_prompt(data)
        steps = [
            {"idx": i, "text": s}
            for i, s in enumerate(data["reasoning"].split("\n\n"))
            if s.strip()
        ]
        prompt = self.judge_template.format(
            PROBLEM=llm_problem,
            STEPS=json.dumps(steps, indent=2),
            FINAL_RESPONSE=data["response"],
        )
        return prompt


@dataclass
class JudgeCoTStepsInResponseTemplate(Template):
    pred_template: PredTemplate
    judge_template_file = JudgeCoTParagraphsTemplate.judge_template_file

    def __post_init__(self):
        self.judge_template = load_template_file(
            templates_root / self.judge_template_file
        )

    def make_prompt(self, data: Mapping) -> str:
        llm_problem = self.pred_template.make_prompt(data)
        steps = extract_key(data, key="steps")
        steps = [{"idx": i, "step": s} for i, s in enumerate(steps)]
        final_response = extract_key(data, key="final_response")
        prompt = self.judge_template.format(
            PROBLEM=llm_problem,
            STEPS=json.dumps(steps, indent=2),
            FINAL_RESPONSE=final_response,
        )
        return prompt


def extract_key(data: Mapping, key: str) -> list[str]:
    if "response" not in data or data["response"] is None or data["response"] == "":
        raise CannotMakePromptError()
    try:
        response = json_repair.loads(data["response"])
    except TypeError as e:
        logger.error(f"Error parsing response: {data['response']}")
        raise CannotMakePromptError() from e
    if key not in response:
        raise CannotMakePromptError()
    if not isinstance(response, dict):
        raise CannotMakePromptError()
    return response[key]


class CannotMakePromptError(Exception):
    pass


@dataclass
class JudgeVerifySolutionTemplate(Template):
    judge_template_file = "judge_verify_solution.txt"

    def __post_init__(self):
        self.judge_template = load_template_file(
            templates_root / self.judge_template_file
        )

    def make_prompt(self, data: Mapping) -> str:
        solution = extract_key(data, key="final_response")
        return self.judge_template.format(
            problem=data["text"],
            reference_solution=data["target"],
            solution=solution,
        )


@dataclass
class JudgeRankingTemplate(Template):
    pred_template: PredTemplate
    judge_template_file = "judge_ranking.txt"

    def __post_init__(self):
        self.judge_template = load_template_file(
            templates_root / self.judge_template_file
        )

    def make_prompt(self, data: Mapping) -> str:
        llm_data = {"text": data["text"][0]}
        llm_problem: str = self.pred_template.make_prompt(llm_data)
        llm_solutions: list[dict] = list(make_llm_solutions(data))
        filled = self.judge_template.format(
            llm_problem=llm_problem, llm_solutions=json.dumps(llm_solutions, indent=2)
        )
        return filled


def make_llm_solutions(data: Mapping) -> Iterable[dict]:
    for i, reasoning, response in zip(
        data["run_idx"], data["reasoning"], data["response"]
    ):
        yield {"run_idx": i, "reasoning": reasoning, "response": response}


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


def make_pred_template(dataset: Dataset, template_id: int) -> PredTemplate:
    if template_id == 33:
        return PredTemplateWithAnnotatorMetadata(
            dataset=dataset, template_id=template_id
        )
    return PredTemplate(dataset=dataset, template_id=template_id)
