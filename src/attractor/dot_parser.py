from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import Edge, Graph, Node
from .utils import derive_class


@dataclass
class Token:
    kind: str
    value: str
    position: int


class DotParseError(ValueError):
    pass


def strip_comments(source: str) -> str:
    result = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch == "\"":
            result.append(ch)
            i += 1
            escaped = False
            while i < len(source):
                result.append(source[i])
                if escaped:
                    escaped = False
                elif source[i] == "\\":
                    escaped = True
                elif source[i] == "\"":
                    i += 1
                    break
                i += 1
            continue
        if source.startswith("//", i):
            while i < len(source) and source[i] != "\n":
                i += 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            if end == -1:
                raise DotParseError("Unterminated block comment")
            i = end + 2
            continue
        result.append(ch)
        i += 1
    return "".join(result)


class Tokenizer:
    def __init__(self, source: str) -> None:
        self.source = strip_comments(source)
        self.length = len(self.source)
        self.index = 0

    def _peek(self) -> str:
        if self.index >= self.length:
            return ""
        return self.source[self.index]

    def _advance(self) -> str:
        ch = self._peek()
        self.index += 1
        return ch

    def tokens(self) -> list[Token]:
        tokens: list[Token] = []
        while self.index < self.length:
            ch = self._peek()
            if ch.isspace():
                self._advance()
                continue
            if ch == "-" and self.source.startswith("->", self.index):
                tokens.append(Token("ARROW", "->", self.index))
                self.index += 2
                continue
            if ch in "{}[]=,;":
                tokens.append(Token(ch, ch, self.index))
                self.index += 1
                continue
            if ch == ".":
                tokens.append(Token(".", ".", self.index))
                self.index += 1
                continue
            if ch == "\"":
                tokens.append(self._read_string())
                continue
            if ch.isdigit() or (ch == "-" and self._peek_number()):
                tokens.append(self._read_number())
                continue
            if ch.isalpha() or ch == "_":
                tokens.append(self._read_identifier())
                continue
            raise DotParseError(f"Unexpected character at {self.index}: {ch}")
        return tokens

    def _peek_number(self) -> bool:
        if self.index + 1 >= self.length:
            return False
        return self.source[self.index + 1].isdigit()

    def _read_string(self) -> Token:
        start = self.index
        self._advance()
        escaped = False
        value_chars: list[str] = []
        while self.index < self.length:
            ch = self._advance()
            if escaped:
                if ch == "n":
                    value_chars.append("\n")
                elif ch == "t":
                    value_chars.append("\t")
                else:
                    value_chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == "\"":
                return Token("STRING", "".join(value_chars), start)
            value_chars.append(ch)
        raise DotParseError("Unterminated string literal")

    def _read_number(self) -> Token:
        start = self.index
        value_chars: list[str] = []
        if self._peek() == "-":
            value_chars.append(self._advance())
        while self.index < self.length and self._peek().isdigit():
            value_chars.append(self._advance())
        if self.index < self.length and self._peek() == ".":
            value_chars.append(self._advance())
            while self.index < self.length and self._peek().isdigit():
                value_chars.append(self._advance())
            return Token("FLOAT", "".join(value_chars), start)
        # duration suffix
        suffix = ""
        if self.index + 1 <= self.length:
            if self.source.startswith("ms", self.index):
                suffix = "ms"
            elif self.source.startswith("s", self.index):
                suffix = "s"
            elif self.source.startswith("m", self.index):
                suffix = "m"
            elif self.source.startswith("h", self.index):
                suffix = "h"
            elif self.source.startswith("d", self.index):
                suffix = "d"
        if suffix:
            self.index += len(suffix)
            return Token("DURATION", "".join(value_chars) + suffix, start)
        return Token("INT", "".join(value_chars), start)

    def _read_identifier(self) -> Token:
        start = self.index
        value_chars: list[str] = []
        while self.index < self.length:
            ch = self._peek()
            if ch.isalnum() or ch == "_":
                value_chars.append(self._advance())
            else:
                break
        return Token("IDENT", "".join(value_chars), start)


@dataclass
class Scope:
    node_defaults: dict[str, Any]
    edge_defaults: dict[str, Any]
    classes: list[str]


def parse_dot(source: str) -> Graph:
    tokens = Tokenizer(source).tokens()
    parser = Parser(tokens)
    return parser.parse()


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.index = 0

    def _peek(self) -> Token | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _advance(self) -> Token | None:
        token = self._peek()
        if token is not None:
            self.index += 1
        return token

    def _expect(self, kind: str, value: str | None = None) -> Token:
        token = self._advance()
        if token is None:
            raise DotParseError(f"Expected {kind} but reached end of input")
        if token.kind != kind:
            raise DotParseError(f"Expected {kind} at {token.position}, got {token.kind}")
        if value is not None and token.value != value:
            raise DotParseError(
                f"Expected {kind} {value} at {token.position}, got {token.value}"
            )
        return token

    def _match(self, kind: str, value: str | None = None) -> bool:
        token = self._peek()
        if token is None:
            return False
        if token.kind != kind:
            return False
        if value is not None and token.value != value:
            return False
        self.index += 1
        return True

    def parse(self) -> Graph:
        token = self._advance()
        if token is None or token.kind != "IDENT" or token.value != "digraph":
            raise DotParseError("Expected 'digraph'")
        graph_id = self._expect("IDENT").value
        graph = Graph(id=graph_id)
        self._expect("{")
        scope = Scope(node_defaults={}, edge_defaults={}, classes=[])
        self._parse_statements(graph, scope, subgraph_nodes=None)
        self._expect("}")
        if self._peek() is not None:
            raise DotParseError("Unexpected tokens after end of graph")
        return graph

    def _parse_statements(
        self,
        graph: Graph,
        scope: Scope,
        subgraph_nodes: list[str] | None,
    ) -> None:
        while True:
            token = self._peek()
            if token is None:
                return
            if token.kind == "}":
                return
            if token.kind == "IDENT" and token.value == "graph":
                self._advance()
                attrs = self._parse_attr_block()
                for key, value in attrs.items():
                    if subgraph_nodes is None:
                        graph.attrs[key] = value
                self._consume_semicolon()
                continue
            if token.kind == "IDENT" and token.value == "node":
                self._advance()
                attrs = self._parse_attr_block()
                scope.node_defaults = {**scope.node_defaults, **attrs}
                self._consume_semicolon()
                continue
            if token.kind == "IDENT" and token.value == "edge":
                self._advance()
                attrs = self._parse_attr_block()
                scope.edge_defaults = {**scope.edge_defaults, **attrs}
                self._consume_semicolon()
                continue
            if token.kind == "IDENT" and token.value == "subgraph":
                self._advance()
                if self._peek() and self._peek().kind == "IDENT":
                    self._advance()
                self._expect("{")
                child_scope = Scope(
                    node_defaults=dict(scope.node_defaults),
                    edge_defaults=dict(scope.edge_defaults),
                    classes=list(scope.classes),
                )
                local_nodes: list[str] = []
                label = self._parse_subgraph(graph, child_scope, local_nodes)
                self._expect("}")
                if label:
                    cls = derive_class(label)
                    if cls:
                        for node_id in local_nodes:
                            node = graph.node(node_id)
                            if node:
                                node.add_class(cls)
                if subgraph_nodes is not None:
                    subgraph_nodes.extend(local_nodes)
                continue
            if token.kind == "IDENT":
                self._parse_statement(graph, scope, subgraph_nodes)
                self._consume_semicolon()
                continue
            raise DotParseError(f"Unexpected token {token.kind} at {token.position}")

    def _parse_subgraph(
        self, graph: Graph, scope: Scope, local_nodes: list[str]
    ) -> str | None:
        label: str | None = None
        while True:
            token = self._peek()
            if token is None or token.kind == "}":
                break
            if token.kind == "IDENT" and token.value == "graph":
                self._advance()
                attrs = self._parse_attr_block()
                if "label" in attrs:
                    label = str(attrs["label"])
                self._consume_semicolon()
                continue
            if token.kind == "IDENT" and token.value == "node":
                self._advance()
                attrs = self._parse_attr_block()
                scope.node_defaults = {**scope.node_defaults, **attrs}
                self._consume_semicolon()
                continue
            if token.kind == "IDENT" and token.value == "edge":
                self._advance()
                attrs = self._parse_attr_block()
                scope.edge_defaults = {**scope.edge_defaults, **attrs}
                self._consume_semicolon()
                continue
            if token.kind == "IDENT" and token.value == "subgraph":
                self._advance()
                if self._peek() and self._peek().kind == "IDENT":
                    self._advance()
                self._expect("{")
                child_scope = Scope(
                    node_defaults=dict(scope.node_defaults),
                    edge_defaults=dict(scope.edge_defaults),
                    classes=list(scope.classes),
                )
                child_nodes: list[str] = []
                child_label = self._parse_subgraph(graph, child_scope, child_nodes)
                self._expect("}")
                if child_label:
                    cls = derive_class(child_label)
                    if cls:
                        for node_id in child_nodes:
                            node = graph.node(node_id)
                            if node:
                                node.add_class(cls)
                local_nodes.extend(child_nodes)
                continue
            if token.kind == "IDENT":
                if self._is_attr_decl():
                    key = self._advance().value
                    self._expect("=")
                    value = self._parse_value()
                    if key == "label":
                        label = str(value)
                    self._consume_semicolon()
                    continue
                self._parse_statement(graph, scope, local_nodes)
                self._consume_semicolon()
                continue
            raise DotParseError(f"Unexpected token {token.kind} at {token.position}")
        return label

    def _is_attr_decl(self) -> bool:
        if self.index + 1 >= len(self.tokens):
            return False
        return self.tokens[self.index + 1].kind == "="

    def _parse_statement(
        self, graph: Graph, scope: Scope, subgraph_nodes: list[str] | None
    ) -> None:
        ident = self._expect("IDENT").value
        if self._match("="):
            value = self._parse_value()
            if subgraph_nodes is None:
                graph.attrs[ident] = value
            return
        if self._match("ARROW"):
            chain = [ident]
            chain.append(self._expect("IDENT").value)
            while self._match("ARROW"):
                chain.append(self._expect("IDENT").value)
            attrs = self._parse_attr_block(optional=True)
            for start, end in zip(chain, chain[1:], strict=False):
                edge_attrs = {**scope.edge_defaults, **attrs}
                graph.add_edge(Edge(from_node=start, to_node=end, attrs=edge_attrs))
            return
        attrs = self._parse_attr_block(optional=True)
        node_attrs = {**scope.node_defaults, **attrs}
        if "label" not in node_attrs:
            node_attrs["label"] = ident
        node = graph.node(ident)
        if node is None:
            node = Node(id=ident, attrs=node_attrs, explicit_attrs=set(attrs.keys()))
            graph.add_node(node)
        else:
            node.attrs.update(node_attrs)
            node.explicit_attrs.update(attrs.keys())
        if subgraph_nodes is not None:
            subgraph_nodes.append(ident)

    def _parse_attr_block(self, optional: bool = False) -> dict[str, Any]:
        if not self._match("["):
            if optional:
                return {}
            raise DotParseError("Expected '[' to start attribute block")
        attrs: dict[str, Any] = {}
        first = True
        while True:
            if self._match("]"):
                break
            if not first:
                if not self._match(","):
                    raise DotParseError("Expected ',' between attributes")
            key = self._parse_key()
            self._expect("=")
            value = self._parse_value()
            attrs[key] = value
            first = False
            if self._match("]"):
                break
        return attrs

    def _parse_key(self) -> str:
        key = self._expect("IDENT").value
        while self._match("."):
            key += "." + self._expect("IDENT").value
        return key

    def _parse_value(self) -> Any:
        token = self._advance()
        if token is None:
            raise DotParseError("Expected value")
        if token.kind == "STRING":
            return token.value
        if token.kind == "INT":
            return int(token.value)
        if token.kind == "FLOAT":
            return float(token.value)
        if token.kind == "DURATION":
            return token.value
        if token.kind == "IDENT":
            if token.value == "true":
                return True
            if token.value == "false":
                return False
            return token.value
        raise DotParseError(f"Unexpected token {token.kind} for value")

    def _consume_semicolon(self) -> None:
        self._match(";")
