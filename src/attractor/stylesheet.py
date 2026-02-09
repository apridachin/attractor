from __future__ import annotations

import re
from dataclasses import dataclass

from .model import Node


@dataclass
class StylesheetRule:
    selector: str
    selector_type: str
    declarations: dict[str, str]
    order: int

    def specificity(self) -> int:
        if self.selector_type == "universal":
            return 0
        if self.selector_type == "shape":
            return 1
        if self.selector_type == "class":
            return 2
        if self.selector_type == "id":
            return 3
        return 0


class StylesheetError(ValueError):
    pass


SELECTOR_RE = re.compile(r"^(\*|#[A-Za-z_][A-Za-z0-9_]*|\.[a-z0-9-]+|[A-Za-z_][A-Za-z0-9_]*)$")


def parse_stylesheet(stylesheet: str) -> list[StylesheetRule]:
    if not stylesheet:
        return []
    rules: list[StylesheetRule] = []
    index = 0
    order = 0
    length = len(stylesheet)
    while index < length:
        if stylesheet[index].isspace():
            index += 1
            continue
        selector_end = stylesheet.find("{", index)
        if selector_end == -1:
            raise StylesheetError("Missing '{' in stylesheet")
        selector = stylesheet[index:selector_end].strip()
        if not SELECTOR_RE.match(selector):
            raise StylesheetError(f"Invalid selector: {selector}")
        selector_type = "universal"
        if selector.startswith("#"):
            selector_type = "id"
        elif selector.startswith("."):
            selector_type = "class"
        elif selector == "*":
            selector_type = "universal"
        else:
            selector_type = "shape"
        index = selector_end + 1
        block_end = stylesheet.find("}", index)
        if block_end == -1:
            raise StylesheetError("Missing '}' in stylesheet")
        block = stylesheet[index:block_end]
        declarations = _parse_declarations(block)
        rules.append(
            StylesheetRule(
                selector=selector,
                selector_type=selector_type,
                declarations=declarations,
                order=order,
            )
        )
        order += 1
        index = block_end + 1
    return rules


def _parse_declarations(block: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    parts = [part.strip() for part in block.split(";") if part.strip()]
    for part in parts:
        if ":" not in part:
            raise StylesheetError(f"Invalid declaration: {part}")
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("\"") and value.endswith("\""):
            value = value[1:-1]
        if key not in {"llm_model", "llm_provider", "reasoning_effort"}:
            raise StylesheetError(f"Unknown property: {key}")
        declarations[key] = value
    return declarations


def apply_stylesheet(rules: list[StylesheetRule], nodes: list[Node]) -> None:
    for node in nodes:
        classes = node.classes()
        best: dict[str, tuple[int, int, str]] = {}
        for rule in rules:
            if not _matches(rule, node, classes):
                continue
            spec = rule.specificity()
            for prop, value in rule.declarations.items():
                if prop in node.explicit_attrs:
                    continue
                current = best.get(prop)
                if current is None or spec > current[0] or (spec == current[0] and rule.order > current[1]):
                    best[prop] = (spec, rule.order, value)
        for prop, (_, __, value) in best.items():
            node.attrs[prop] = value


def _matches(rule: StylesheetRule, node: Node, classes: list[str]) -> bool:
    if rule.selector_type == "universal":
        return True
    if rule.selector_type == "id":
        return rule.selector[1:] == node.id
    if rule.selector_type == "class":
        return rule.selector[1:] in classes
    if rule.selector_type == "shape":
        return rule.selector == node.shape
    return False
