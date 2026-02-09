from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .context import Context
from .outcome import Outcome

CLAUSE_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_.]*)\s*(?P<op>!=|=)?\s*(?P<value>.*)$")


@dataclass
class ConditionError(Exception):
    message: str


@dataclass
class Clause:
    key: str
    op: str
    literal: str


def parse_condition(condition: str) -> list[Clause]:
    clauses: list[Clause] = []
    if not condition:
        return clauses
    for raw in condition.split("&&"):
        raw = raw.strip()
        if not raw:
            continue
        match = CLAUSE_RE.match(raw)
        if not match:
            raise ConditionError(f"Invalid condition clause: {raw}")
        key = match.group("key")
        op = match.group("op") or ""
        value = match.group("value").strip()
        if not op:
            clauses.append(Clause(key=key, op="", literal=""))
            continue
        if value == "":
            raise ConditionError(f"Missing literal for clause: {raw}")
        literal = strip_quotes(value)
        clauses.append(Clause(key=key, op=op, literal=literal))
    return clauses


def strip_quotes(value: str) -> str:
    value = value.strip()
    if value.startswith("\"") and value.endswith("\"") and len(value) >= 2:
        return value[1:-1]
    return value


def resolve_key(key: str, outcome: Outcome, context: Context) -> str:
    if key == "outcome":
        return outcome.status.value
    if key == "preferred_label":
        return outcome.preferred_label or ""
    if key.startswith("context."):
        value = context.get(key)
        if value is None:
            value = context.get(key[len("context.") :])
        return normalize_value(value)
    value = context.get(key)
    return normalize_value(value)


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def evaluate_condition(condition: str, outcome: Outcome, context: Context) -> bool:
    if not condition:
        return True
    clauses = parse_condition(condition)
    for clause in clauses:
        if clause.op == "":
            if not resolve_key(clause.key, outcome, context):
                return False
            continue
        left = resolve_key(clause.key, outcome, context)
        right = clause.literal
        if clause.op == "=":
            if left != right:
                return False
        elif clause.op == "!=":
            if left == right:
                return False
        else:
            raise ConditionError(f"Unsupported operator: {clause.op}")
    return True


def validate_condition(condition: str) -> str | None:
    if not condition:
        return None
    try:
        parse_condition(condition)
    except ConditionError as exc:
        return exc.message
    return None
