from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    FREEFORM = "freeform"
    CONFIRMATION = "confirmation"
    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    FREE_TEXT = "free_text"
    CONFIRM = "confirm"


class AnswerValue(str, Enum):
    YES = "yes"
    NO = "no"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class Option:
    key: str
    label: str


@dataclass
class Question:
    text: str
    type: QuestionType
    options: list[Option] = field(default_factory=list)
    default: Answer | None = None
    timeout_seconds: float | None = None
    stage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    value: str | None = None
    selected_option: Option | None = None
    text: str = ""


class Interviewer:
    def ask(self, question: Question) -> Answer:
        raise NotImplementedError

    def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        return [self.ask(q) for q in questions]

    def inform(self, message: str, stage: str) -> None:
        return None


class AutoApproveInterviewer(Interviewer):
    def ask(self, question: Question) -> Answer:
        if question.type in {QuestionType.YES_NO, QuestionType.CONFIRMATION, QuestionType.CONFIRM}:
            return Answer(value=AnswerValue.YES.value)
        if question.options:
            option = question.options[0]
            return Answer(value=option.key, selected_option=option)
        return Answer(value="auto-approved", text="auto-approved")


class ConsoleInterviewer(Interviewer):
    def ask(self, question: Question) -> Answer:
        print(f"[?] {question.text}")
        if question.options:
            for option in question.options:
                print(f"  [{option.key}] {option.label}")
            response = input("Select: ").strip()
            return _match_option(response, question.options)
        if question.type in {QuestionType.YES_NO, QuestionType.CONFIRMATION, QuestionType.CONFIRM}:
            response = input("[Y/N]: ").strip().lower()
            return Answer(value=AnswerValue.YES.value if response == "y" else AnswerValue.NO.value)
        response = input("> ").strip()
        return Answer(text=response, value=response)


class CallbackInterviewer(Interviewer):
    def __init__(self, callback: Callable[[Question], Answer]) -> None:
        self.callback = callback

    def ask(self, question: Question) -> Answer:
        return self.callback(question)


class QueueInterviewer(Interviewer):
    def __init__(self, answers: Iterable[Answer]) -> None:
        self._queue: list[Answer] = list(answers)

    def ask(self, question: Question) -> Answer:
        if self._queue:
            return self._queue.pop(0)
        return Answer(value=AnswerValue.SKIPPED.value)


class RecordingInterviewer(Interviewer):
    def __init__(self, inner: Interviewer) -> None:
        self.inner = inner
        self.recordings: list[tuple[Question, Answer]] = []

    def ask(self, question: Question) -> Answer:
        answer = self.inner.ask(question)
        self.recordings.append((question, answer))
        return answer


def _match_option(response: str, options: list[Option]) -> Answer:
    response = response.strip()
    for option in options:
        if option.key.lower() == response.lower():
            return Answer(value=option.key, selected_option=option)
        if option.label.lower() == response.lower():
            return Answer(value=option.key, selected_option=option)
    if options:
        return Answer(value=options[0].key, selected_option=options[0])
    return Answer(value=AnswerValue.SKIPPED.value)
