"""Dataset loading utilities for evaluation pipelines.

This module implements a simple strategy pattern so that additional datasets can
be plugged into the evaluation pipeline without modifying the orchestration
code.  Each loader is responsible for converting raw data into a
:class:`DatasetBundle`, which contains the entries handed off to the prompt
processor as well as the aggregate human statistics used for meta-evaluation."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Type

import itertools
import re

TEXT_EXTENSIONS = {
    ".c",
    ".h",
    ".py",
    ".s",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".sh",
    ".cpp",
    ".hpp",
    ".java",
    ".rs",
    ".go",
    ".js",
    ".ts",
}


@dataclass
class DatasetBundle:
    """Container returned by dataset loaders."""

    entries: List[dict]
    human_means: List[float]


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield JSON objects from a JSON lines file."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


class DatasetLoader(ABC):
    """Strategy interface for dataset ingestion."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @abstractmethod
    def load(self) -> DatasetBundle:
        """Return the dataset bundle for the configured root."""


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.name.lower() == "makefile"


def _read_file_safely(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logging.debug("Skipping non-text file: %s", path)
        return None




def _load_json_data(path: Path, description: str) -> object:
    """Return parsed JSON content while surfacing the failing file."""

    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {description}: {exc}") from exc


def _mean_or_zero(values: Sequence[float]) -> float:
    """Return the arithmetic mean or zero when inputs are empty."""

    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _collect_context_files(root: Path) -> List[Dict[str, str]]:
    """Collect text files under ``root`` along with relative paths and contents."""

    materials: List[Dict[str, str]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and _is_text_file(candidate):
            content = _read_file_safely(candidate)
            if content is not None:
                materials.append(
                    {
                        "relative_path": str(candidate.relative_to(root)),
                        "content": content,
                    }
                )
    return materials


class LLMGraderDatasetLoader(DatasetLoader):
    """Load the entire llm_grader dataset, including tutorial code context."""

    def __init__(self, root: Path, *, max_examples: Optional[int] = None) -> None:
        super().__init__(root)
        self.max_examples = max_examples

    def load(self) -> DatasetBundle:
        dataset_root = self.root
        grading_dirs = sorted(
            p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("q")
        )

        entries: List[dict] = []
        human_means: List[float] = []

        for question_dir in grading_dirs:
            question_id = question_dir.name
            grading_path = question_dir / "grading.json"
            tutorial_code_dir = dataset_root / "tutorialCode" / question_id
            tutorial_criteria_path = dataset_root / "tutorialCriteria" / f"{question_id}.json"

            submissions_payload = _load_json_data(grading_path, "llm_grader grading data")
            criteria_payload_raw = _load_json_data(
                tutorial_criteria_path,
                "llm_grader tutorial criteria",
            )

            if not isinstance(submissions_payload, dict):
                logging.debug("Unexpected submissions format for %s", grading_path)
                continue
            if not isinstance(criteria_payload_raw, dict):
                logging.debug("Unexpected tutorial criteria format for %s", tutorial_criteria_path)
                continue

            criteria_payload = next(iter(criteria_payload_raw.values()), {})
            instructor_criteria = criteria_payload.get("criteria", "")

            context_files = _collect_context_files(tutorial_code_dir)

            for submission_id, payload in submissions_payload.items():
                if self.max_examples is not None and len(entries) >= self.max_examples:
                    break

                if not isinstance(payload, dict):
                    logging.debug("Skipping submission %s due to malformed payload", submission_id)
                    continue

                sample = next(iter(payload.values()), {})
                if not isinstance(sample, dict):
                    logging.debug("Skipping submission %s due to malformed sample", submission_id)
                    continue

                human_scores = [
                    sample.get(f"score_{i}")
                    for i in range(1, 4)
                    if sample.get(f"score_{i}") is not None
                ]
                human_mean = _mean_or_zero(human_scores)
                human_means.append(human_mean)

                entry = {
                    "id": f"{question_id}-{submission_id}",
                    "question_id": question_id,
                    "question": sample.get("question", []),
                    "answer": sample.get("answer", ""),
                    "sample_answer": sample.get("sample_answer", ""),
                    "sample_criteria": sample.get("sample_criteria", ""),
                    "tutorial_criteria": instructor_criteria,
                    "max_score": sample.get("full_points", criteria_payload.get("full_points")),
                    "human_scores": human_scores,
                    "context_files": [dict(context) for context in context_files],
                }
                entries.append(entry)

            if self.max_examples is not None and len(entries) >= self.max_examples:
                break

        logging.info(
            "Loaded %d llm_grader submissions across %d questions",
            len(entries),
            len(grading_dirs),
        )
        return DatasetBundle(entries=entries, human_means=human_means)


class BiGGenDatasetLoader(DatasetLoader):
    """Load BiGGen-Bench response records for pointwise evaluation."""

    DEFAULT_RESPONSES = "sample_responses.json"
    DEFAULT_EVALUATIONS = "sample_evals.json"

    def __init__(
        self,
        root: Path,
        *,
        responses_filename: Optional[str] = None,
        evaluations_filename: Optional[str] = None,
        max_examples: Optional[int] = None,
    ) -> None:
        super().__init__(root)
        self.responses_path = root / (responses_filename or self.DEFAULT_RESPONSES)
        self.evaluations_path = (
            root / evaluations_filename if evaluations_filename else root / self.DEFAULT_EVALUATIONS
        )
        self.max_examples = max_examples

    def _load_optional_evaluations(self) -> Dict[str, dict]:
        if not self.evaluations_path.exists():
            logging.warning(
                "BiGGen-Bench evaluations file '%s' not found; human scores unavailable.",
                self.evaluations_path,
            )
            return {}
        try:
            with self.evaluations_path.open(encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            logging.error("Failed to parse BiGGen-Bench evaluations: %s", exc)
            return {}
        if not isinstance(payload, dict):
            logging.warning("Unexpected format for BiGGen-Bench evaluations file.")
            return {}
        return payload

    @staticmethod
    def _extract_human_scores(record: dict) -> List[float]:
        scores: List[float] = []
        candidate_items = [
            record.get("human_score"),
            record.get("human_scores"),
            record.get("avg_human_score"),
            record.get("average_human_score"),
            record.get("human_judgment"),
        ]
        for item in candidate_items:
            if item is None:
                continue
            if isinstance(item, (int, float)):
                scores.append(float(item))
            elif isinstance(item, list):
                for value in item:
                    if isinstance(value, (int, float)):
                        scores.append(float(value))
        return scores

    def load(self) -> DatasetBundle:
        if not self.responses_path.exists():
            raise FileNotFoundError(
                f"BiGGen-Bench responses file '{self.responses_path}' not found."
            )

        with self.responses_path.open(encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, dict):
            raise ValueError("Expected BiGGen-Bench responses to be a JSON object keyed by id.")

        evaluations = self._load_optional_evaluations()
        entries: List[dict] = []
        human_means: List[float] = []
        proxy_count = 0

        for idx, key in enumerate(sorted(raw_data.keys())):
            if self.max_examples is not None and idx >= self.max_examples:
                break

            sample = raw_data[key]
            if not isinstance(sample, dict):
                logging.debug("Skipping malformed BiGGen-Bench entry '%s'", key)
                continue

            evaluation_record = evaluations.get(key, {}) if isinstance(evaluations, dict) else {}
            human_scores = self._extract_human_scores(evaluation_record)
            score_source = "human_annotations" if human_scores else "missing"
            score_provider: Optional[str] = None
            if not human_scores and "score" in evaluation_record:
                value = evaluation_record["score"]
                if isinstance(value, (int, float)):
                    logging.debug(
                        "Treating evaluation score for '%s' as proxy human score due to missing human annotations.",
                        key,
                    )
                    human_scores = [float(value)]
                    score_source = "proxy_gpt4.1_score"
                    score_provider = "gpt-4.1"
                    proxy_count += 1

            if human_scores:
                human_mean = sum(human_scores) / len(human_scores)
            else:
                human_mean = 0.0

            score_rubric = sample.get("score_rubric", {}) or {}
            entry = {
                "id": sample.get("id", key),
                "input": sample.get("input", ""),
                "response": sample.get("response", ""),
                "reference_answer": sample.get("reference_answer"),
                "score_rubric": score_rubric,
                "capability": sample.get("capability"),
                "task": sample.get("task"),
                "instance_idx": sample.get("instance_idx"),
                "system_prompt": sample.get("system_prompt"),
                "max_score": 5,
                "human_scores": human_scores,
                "human_score_source": score_source,
                "human_score_provider": score_provider,
            }

            entries.append(entry)
            human_means.append(human_mean)

        logging.info(
            "Loaded %d BiGGen-Bench examples from %s",
            len(entries),
            self.responses_path,
        )
        if not any(entry.get("human_scores") for entry in entries):
            logging.warning(
                "No human scores detected for BiGGen-Bench; correlation metrics will use zeros as placeholders."
            )
        elif proxy_count:
            logging.warning(
                "Using evaluation scores as proxy human annotations for %d BiGGen-Bench entries.",
                proxy_count,
            )
        return DatasetBundle(entries=entries, human_means=human_means)


class FLASKDatasetLoader(DatasetLoader):
    """Load FLASK evaluation examples and optional model responses."""

    DEFAULT_SPLIT = "standard"
    DEFAULT_EVAL_PATH = "evaluation_set/flask_evaluation.jsonl"
    DEFAULT_HARD_PATH = "evaluation_set/flask_hard_evaluation.jsonl"
    DEFAULT_SKILL_METADATA = "metadata_annotation/skillset/src/skillset_description.json"
    DEFAULT_RESPONSE_PATH = "model_output/outputs/gpt4.jsonl"
    DEFAULT_REVIEW_PATH = "gpt_review/outputs/gpt4_review.jsonl"

    def __init__(
        self,
        root: Path,
        *,
        split: str = DEFAULT_SPLIT,
        eval_path: Optional[Path] = None,
        hard_path: Optional[Path] = None,
        skill_metadata_path: Optional[Path] = None,
        response_path: Optional[Path] = None,
        review_path: Optional[Path] = None,
        max_examples: Optional[int] = None,
    ) -> None:
        super().__init__(root)
        self.split = split.lower()
        self.eval_path = (
            eval_path if eval_path is not None else self.root / self.DEFAULT_EVAL_PATH
        )
        self.hard_path = (
            hard_path if hard_path is not None else self.root / self.DEFAULT_HARD_PATH
        )
        self.skill_metadata_path = (
            skill_metadata_path
            if skill_metadata_path is not None
            else self.root / self.DEFAULT_SKILL_METADATA
        )
        self.response_path = (
            response_path
            if response_path is not None
            else self.root / self.DEFAULT_RESPONSE_PATH
        )
        self.review_path = (
            review_path
            if review_path is not None
            else self.root / self.DEFAULT_REVIEW_PATH
        )
        self.max_examples = max_examples
        self._skill_metadata: Dict[str, dict] | None = None
        self._response_lookup: Dict[str, str] | None = None
        self._review_scores: Dict[str, List[float]] | None = None

    def _select_eval_file(self) -> Path:
        if self.split == "hard":
            if not self.hard_path.exists():
                raise FileNotFoundError(
                    f"FLASK hard split not found at '{self.hard_path}'."
                )
            return self.hard_path
        if not self.eval_path.exists():
            raise FileNotFoundError(
                f"FLASK evaluation set not found at '{self.eval_path}'."
            )
        return self.eval_path

    def _load_review_scores(self) -> Dict[str, List[float]]:
        if self._review_scores is not None:
            return self._review_scores

        scores: Dict[str, List[float]] = {}
        if not self.review_path.exists():
            logging.warning(
                "FLASK review scores file '%s' not found; human alignment metrics will be skipped.",
                self.review_path,
            )
            self._review_scores = scores
            return scores

        for record in iter_jsonl(self.review_path):
            question_id = record.get("question_id")
            if question_id is None:
                continue
            raw_scores = record.get("score") or record.get("scores")
            extracted: List[float] = []
            if isinstance(raw_scores, dict):
                for value in raw_scores.values():
                    if isinstance(value, (int, float)):
                        extracted.append(float(value))
            elif isinstance(raw_scores, list):
                for value in raw_scores:
                    if isinstance(value, (int, float)):
                        extracted.append(float(value))
            if not extracted:
                continue
            scores[str(question_id)] = extracted

        if scores:
            logging.info(
                "Loaded FLASK review scores for %d instances from %s",
                len(scores),
                self.review_path,
            )
        else:
            logging.warning(
                "FLASK review score file '%s' contained no numeric entries; proceeding without human comparisons.",
                self.review_path,
            )

        self._review_scores = scores
        return scores

    def _load_skill_metadata(self) -> Dict[str, dict]:
        if self._skill_metadata is not None:
            return self._skill_metadata
        if not self.skill_metadata_path.exists():
            raise FileNotFoundError(
                f"FLASK skill metadata not found at '{self.skill_metadata_path}'."
            )
        with self.skill_metadata_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        metadata: Dict[str, dict] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            name = record.get("Skill")
            criteria = record.get("Criteria")
            scoring = record.get("Scoring")
            if not isinstance(name, str) or not isinstance(criteria, str):
                continue
            if not isinstance(scoring, dict):
                scoring = {}
            metadata[name] = {
                "name": name,
                "criteria": criteria.strip(),
                "scale": {
                    str(level): str(description)
                    for level, description in scoring.items()
                },
            }
        self._skill_metadata = metadata
        return metadata

    def _load_responses(self) -> Dict[str, str]:
        if self._response_lookup is not None:
            return self._response_lookup
        responses: Dict[str, str] = {}
        if self.response_path.exists():
            for record in iter_jsonl(self.response_path):
                question_id = record.get("question_id")
                text = record.get("text")
                if question_id is None or not isinstance(text, str):
                    continue
                responses[str(question_id)] = text.strip()
        else:
            logging.warning(
                "FLASK response path '%s' not found; proceeding without model outputs.",
                self.response_path,
            )
        self._response_lookup = responses
        return responses

    def _build_rubric(self, skills: Iterable[str]) -> dict:
        metadata = self._load_skill_metadata()
        skill_sections: List[dict] = []
        for skill_name in skills:
            info = metadata.get(skill_name)
            if info is None:
                logging.debug("Unknown FLASK skill encountered: %s", skill_name)
                continue
            skill_sections.append(info)
        rubric = {
            "skills": skill_sections,
            "scale_min": 1,
            "scale_max": 5,
        }
        return rubric

    def load(self) -> DatasetBundle:
        eval_file = self._select_eval_file()
        responses = self._load_responses()
        review_scores = self._load_review_scores()

        entries: List[dict] = []
        human_means: List[float] = []

        iterator: Iterable[dict] = iter_jsonl(eval_file)
        if self.max_examples is not None:
            iterator = itertools.islice(iterator, self.max_examples)

        for record in iterator:
            idx = record.get("idx")
            instruction = record.get("instruction") or ""
            reference_answer = record.get("answer")
            skills = record.get("skill") or []
            rubric = self._build_rubric(skills)
            response_text = responses.get(str(idx), "")
            if not response_text:
                logging.debug("No model response available for FLASK example %s", idx)

            score_list = review_scores.get(str(idx), [])
            if score_list:
                human_mean = sum(score_list) / len(score_list)
            else:
                human_mean = 0.0

            entry = {
                "id": f"flask-{idx}",
                "instructions": str(instruction).strip(),
                "reference_answer": reference_answer if isinstance(reference_answer, str) else None,
                "score_rubric": rubric,
                "response": response_text,
                "skills": list(skills) if isinstance(skills, list) else [],
                "domain": record.get("domain", []),
                "difficulty": record.get("difficulty"),
                "task": record.get("task"),
                "max_score": 5,
                "human_scores": score_list,
                "human_score_source": "gpt_review" if score_list else "missing",
                "human_score_provider": self.review_path.stem if score_list else None,
            }
            entries.append(entry)
            human_means.append(human_mean)

        scored_examples = sum(1 for entry in entries if entry.get("human_scores"))
        if scored_examples:
            logging.info(
                "Loaded %d FLASK examples from %s with %d review score sets",
                len(entries),
                eval_file,
                scored_examples,
            )
        else:
            logging.warning(
                "Loaded %d FLASK examples from %s but no review scores were found; human alignment metrics will be skipped.",
                len(entries),
                eval_file,
            )
        return DatasetBundle(entries=entries, human_means=human_means)


try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency in CI
    load_dataset = None  # type: ignore


def load_mtbench_hj(data_dir: Optional[str] = None):
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' package is required to load MT-Bench human judgments."
        )
    kwargs = {}
    if data_dir:
        kwargs["data_dir"] = data_dir
    return load_dataset("lmsys/mt_bench_human_judgments", **kwargs)


def load_arena(data_dir: Optional[str] = None):
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' package is required to load Chatbot Arena conversations."
        )
    kwargs = {}
    if data_dir:
        kwargs["data_dir"] = data_dir
    return load_dataset("lmsys/chatbot_arena_conversations", **kwargs)


class PairwiseJudgmentDatasetLoader(DatasetLoader):
    """Base utilities for datasets that provide pairwise human preferences."""

    def __init__(
        self,
        root: Path,
        *,
        data_dir: Optional[str] = None,
        max_examples: Optional[int] = None,
        scoring_scale: int = 10,
        hf_chunk_pct: float = 10.0,
    ) -> None:
        super().__init__(root)
        self.data_dir = data_dir
        self.max_examples = max_examples
        self.scoring_scale = scoring_scale
        self.hf_chunk_pct = hf_chunk_pct

    def _load_dataset(self):
        raise NotImplementedError

    @staticmethod
    def _normalize_winner(value) -> str:
        if value is None:
            return "tie"
        normalized = str(value).strip().lower()
        if normalized in {"a", "model_a", "left", "first", "1"}:
            return "a"
        if normalized in {"b", "model_b", "right", "second", "2"}:
            return "b"
        if normalized in {"tie", "draw", "0", "none", "both"}:
            return "tie"
        return normalized

    def _iter_hf_dataset(self, dataset_id: str):
        if load_dataset is None:
            raise ImportError(
                "The 'datasets' package is required to load dataset '%s'" % dataset_id
            )
        kwargs: Dict[str, str] = {}
        if self.data_dir:
            kwargs["data_dir"] = self.data_dir

        chunk = max(float(self.hf_chunk_pct or 0.0), 0.0)
        if chunk <= 0.0 or chunk >= 100.0:
            yield load_dataset(dataset_id, split="train", **kwargs)
            return

        start = 0.0
        while start < 100.0:
            end = min(100.0, start + chunk)
            split = f"train[{start}%:{end}%]"
            logging.debug("Loading HF split %s for %s", split, dataset_id)
            yield load_dataset(dataset_id, split=split, **kwargs)
            start = end


class MTBenchHumanJudgmentsLoader(PairwiseJudgmentDatasetLoader):
    """Load MT-Bench human pairwise preferences."""

    DEFAULT_LOCAL_PATH = Path(
        "prometheus-eval/eval/benchmark/data/mt_bench_human_judgement_eval.json"
    )
    HF_DATASET_ID = "lmsys/mt_bench_human_judgments"

    def __init__(
        self,
        root: Path,
        *,
        data_dir: Optional[str] = None,
        max_examples: Optional[int] = None,
        scoring_scale: int = 5,
        local_path: Optional[Path] = None,
        hf_chunk_pct: float = 10.0,
    ) -> None:
        super().__init__(
            root,
            data_dir=data_dir,
            max_examples=max_examples,
            scoring_scale=scoring_scale,
            hf_chunk_pct=hf_chunk_pct,
        )
        self.local_path = local_path or (self.root / self.DEFAULT_LOCAL_PATH)

    @staticmethod
    def _extract_section(block: str, header: str, *, next_headers: Sequence[str]) -> str:
        match = re.search(re.escape(header), block)
        if not match:
            return ""
        start = match.end()
        end = len(block)
        for nxt in next_headers:
            idx = block.find(nxt, start)
            if idx != -1:
                end = min(end, idx)
        return block[start:end].strip()

    def _parse_mtbench_record(self, record: dict, idx: int) -> dict:
        headers = [
            "###Task Description:",
            "###The instruction to evaluate:",
            "###Response to evaluate:",
            "###Reference Answer (Score 5):",
            "###Score Rubrics:",
            "###Feedback:",
        ]

        def _sections(text: str) -> dict:
            out: Dict[str, str] = {}
            for pos, header in enumerate(headers):
                out[header] = self._extract_section(
                    text, header, next_headers=headers[pos + 1 :]
                )
            return out

        chosen_sections = _sections(record.get("chosen_instruction", ""))
        rejected_sections = _sections(record.get("rejected_instruction", ""))

        question_text = chosen_sections.get("###The instruction to evaluate:") or ""
        response_a = chosen_sections.get("###Response to evaluate:") or ""
        response_b = rejected_sections.get("###Response to evaluate:") or ""
        rubric_text = chosen_sections.get("###Score Rubrics:") or ""
        human_winner = "tie" if int(record.get("tie", 0)) else "a"

        return {
            "id": f"mt_bench-{idx}",
            "question_id": idx,
            "question": question_text.strip(),
            "model_a": record.get("chosen_model"),
            "model_b": record.get("rejected_model"),
            "response_a": response_a.strip(),
            "response_b": response_b.strip(),
            "human_winner": human_winner,
            "category": record.get("category"),
            "source_dataset": "mt_bench",
            "max_score": self.scoring_scale,
            "score_rubric": {
                "description": rubric_text.strip(),
                "scale_min": 1,
                "scale_max": self.scoring_scale,
            },
        }

    def _load_dataset(self):
        if load_dataset is None:
            raise ImportError(
                "The 'datasets' package is required to load MT-Bench human judgments."
            )
        kwargs: Dict[str, str] = {}
        if self.data_dir:
            kwargs["data_dir"] = self.data_dir
        return load_dataset(self.HF_DATASET_ID, **kwargs)["train"]

    def _load_local(self) -> List[dict]:
        if not self.local_path.exists():
            raise FileNotFoundError(
                "Local MT-Bench human judgments file not found. Provide --mtb_data_dir pointing to a Hugging Face "
                "snapshot directory or place the JSON file at 'prometheus-eval/eval/benchmark/data/mt_bench_human_judgement_eval.json'."
            )
        with self.local_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError("Unexpected MT-Bench local file format (expected a list).")
        iterator: Iterable[dict] = records
        if self.max_examples is not None:
            iterator = itertools.islice(records, self.max_examples)
        entries = [
            self._parse_mtbench_record(record, idx)
            for idx, record in enumerate(iterator)
            if isinstance(record, dict)
        ]
        logging.info("Loaded %d MT-Bench human judgments from local file", len(entries))
        return entries

    def load(self) -> DatasetBundle:
        if self.data_dir:
            entries: List[dict] = []
            processed = 0
            for dataset_chunk in self._iter_hf_dataset(self.HF_DATASET_ID):
                iterator = dataset_chunk
                if self.max_examples is not None:
                    remaining = self.max_examples - processed
                    if remaining <= 0:
                        break
                    iterator = itertools.islice(dataset_chunk, remaining)

                for record in iterator:
                    question_id = record.get("question_id", processed)
                    question_text = record.get("question", "")
                    entry = {
                        "id": f"mt_bench-{question_id}-{processed}",
                        "question_id": question_id,
                        "question": question_text,
                        "model_a": record.get("model_a"),
                        "model_b": record.get("model_b"),
                        "response_a": record.get("answer_a") or "",
                        "response_b": record.get("answer_b") or "",
                        "human_winner": self._normalize_winner(record.get("winner")),
                        "category": record.get("category"),
                        "source_dataset": "mt_bench",
                        "max_score": self.scoring_scale,
                        "score_rubric": {
                            "description": (
                                "Rate the assistant response for overall helpfulness, accuracy, "
                                "and clarity on a {scale}-point scale (higher is better)."
                            ).format(scale=self.scoring_scale),
                            "scale_min": 1,
                            "scale_max": self.scoring_scale,
                        },
                    }
                    entries.append(entry)
                    processed += 1

            logging.info(
                "Loaded %d MT-Bench human judgments from Hugging Face snapshot (chunk size %.1f%%)",
                len(entries),
                self.hf_chunk_pct,
            )
            return DatasetBundle(entries=entries, human_means=[])

        entries = self._load_local()
        return DatasetBundle(entries=entries, human_means=[])


class ChatbotArenaLoader(PairwiseJudgmentDatasetLoader):
    """Load Chatbot Arena human preference comparisons."""

    DEFAULT_LOCAL_PATH = Path("prometheus-eval/eval/benchmark/data/autoj_pairwise.json")
    HF_DATASET_ID = "lmsys/chatbot_arena_conversations"

    @staticmethod
    def _extract_final_response(conversation: Iterable[dict]) -> str:
        final_text = ""
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).lower()
            if role == "assistant":
                text = turn.get("content") or turn.get("text") or ""
                final_text = str(text)
        return final_text.strip()

    def __init__(
        self,
        root: Path,
        *,
        data_dir: Optional[str] = None,
        max_examples: Optional[int] = None,
        scoring_scale: int = 10,
        local_path: Optional[Path] = None,
        hf_chunk_pct: float = 10.0,
    ) -> None:
        super().__init__(
            root,
            data_dir=data_dir,
            max_examples=max_examples,
            scoring_scale=scoring_scale,
            hf_chunk_pct=hf_chunk_pct,
        )
        self.local_path = local_path or (self.root / self.DEFAULT_LOCAL_PATH)

    def _load_dataset(self):
        if load_dataset is None:
            raise ImportError(
                "The 'datasets' package is required to load Chatbot Arena conversations."
            )
        kwargs: Dict[str, str] = {}
        if self.data_dir:
            kwargs["data_dir"] = self.data_dir
        return load_dataset(self.HF_DATASET_ID, **kwargs)["train"]

    def _load_local(self) -> List[dict]:
        if not self.local_path.exists():
            raise FileNotFoundError(
                "Local Chatbot Arena dataset not found. Provide --arena_data_dir pointing to the snapshot directory."
            )
        entries: List[dict] = []
        with self.local_path.open("r", encoding="utf-8") as handle:
            iterator = (json.loads(line) for line in handle if line.strip())
            if self.max_examples is not None:
                iterator = itertools.islice(iterator, self.max_examples)
            for idx, record in enumerate(iterator):
                if not isinstance(record, dict):
                    continue
                label = record.get("label")
                if label == 1:
                    winner = "a"
                elif label == 0:
                    winner = "b"
                else:
                    winner = "tie"
                entries.append(
                    {
                        "id": f"chatbot_arena-local-{idx}",
                        "question": record.get("prompt", ""),
                        "model_a": "response_1",
                        "model_b": "response_2",
                        "response_a": record.get("response 1", ""),
                        "response_b": record.get("response 2", ""),
                        "human_winner": winner,
                        "language": None,
                        "category": record.get("scenario"),
                        "source_dataset": "chatbot_arena",
                        "max_score": self.scoring_scale,
                        "score_rubric": {
                            "description": (
                                "Score the assistant reply considering contextual appropriateness, "
                                "helpfulness, and coherence on a {scale}-point scale."
                            ).format(scale=self.scoring_scale),
                            "scale_min": 1,
                            "scale_max": self.scoring_scale,
                        },
                    }
                )
        logging.info("Loaded %d Chatbot Arena comparisons from local file", len(entries))
        return entries

    def load(self) -> DatasetBundle:
        if self.data_dir:
            entries: List[dict] = []
            processed = 0
            for dataset_chunk in self._iter_hf_dataset(self.HF_DATASET_ID):
                iterator = dataset_chunk
                if self.max_examples is not None:
                    remaining = self.max_examples - processed
                    if remaining <= 0:
                        break
                    iterator = itertools.islice(dataset_chunk, remaining)

                for record in iterator:
                    conversation_a = record.get("conversation_a", [])
                    conversation_b = record.get("conversation_b", [])
                    entry = {
                        "id": f"chatbot_arena-{record.get('conversation_id', processed)}",
                        "conversation_id": record.get("conversation_id"),
                        "question": record.get("question", ""),
                        "model_a": record.get("model_a"),
                        "model_b": record.get("model_b"),
                        "response_a": self._extract_final_response(conversation_a),
                        "response_b": self._extract_final_response(conversation_b),
                        "conversation_a": list(conversation_a),
                        "conversation_b": list(conversation_b),
                        "human_winner": self._normalize_winner(record.get("winner")),
                        "language": record.get("language"),
                        "source_dataset": "chatbot_arena",
                        "max_score": self.scoring_scale,
                        "score_rubric": {
                            "description": (
                                "Score the assistant reply considering contextual appropriateness, "
                                "helpfulness, safety, and coherence on a {scale}-point scale."
                            ).format(scale=self.scoring_scale),
                            "scale_min": 1,
                            "scale_max": self.scoring_scale,
                        },
                    }
                    entries.append(entry)
                    processed += 1

            logging.info(
                "Loaded %d Chatbot Arena comparisons from Hugging Face snapshot (chunk size %.1f%%)",
                len(entries),
                self.hf_chunk_pct,
            )
            return DatasetBundle(entries=entries, human_means=[])

        entries = self._load_local()
        return DatasetBundle(entries=entries, human_means=[])


DATASET_LOADERS: Dict[str, Type[DatasetLoader]] = {
    "llm_grader": LLMGraderDatasetLoader,
    "biggen_bench": BiGGenDatasetLoader,
    "flask": FLASKDatasetLoader,
    "mt_bench": MTBenchHumanJudgmentsLoader,
    "chatbot_arena": ChatbotArenaLoader,
}


def register_dataset_loader(name: str, loader_cls: Type[DatasetLoader]) -> None:
    DATASET_LOADERS[name] = loader_cls


def create_dataset_loader(name: str, root: Path, **kwargs) -> DatasetLoader:
    try:
        loader_cls = DATASET_LOADERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset loader '{name}'") from exc
    return loader_cls(root, **kwargs)


def load_llm_grader_dataset(root: Path) -> DatasetBundle:
    """Convenience wrapper for existing callers."""

    return LLMGraderDatasetLoader(root).load()
