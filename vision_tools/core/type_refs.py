from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


def _split_top_level(value: str, separator: str) -> list[str]:
    parts: list[str] = []
    bracket_depth = 0
    current: list[str] = []

    for char in value:
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1

        if char == separator and bracket_depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


@dataclass(frozen=True)
class PortTypeRef:
    def __str__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class SimpleTypeRef(PortTypeRef):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class GenericTypeRef(PortTypeRef):
    name: str
    item_type: PortTypeRef

    def __str__(self) -> str:
        return f"{self.name}[{self.item_type}]"


@dataclass(frozen=True)
class UnionTypeRef(PortTypeRef):
    options: tuple[PortTypeRef, ...]

    def __str__(self) -> str:
        return " | ".join(str(option) for option in self.options)


def parse_type_ref(value: str | PortTypeRef) -> PortTypeRef:
    if isinstance(value, PortTypeRef):
        return value

    text = value.strip()
    if not text:
        raise ValueError("Type ref cannot be empty.")

    union_parts = _split_top_level(text, "|")
    if len(union_parts) > 1:
        return UnionTypeRef(tuple(parse_type_ref(part) for part in union_parts))

    if text.endswith("]") and "[" in text:
        name, remainder = text.split("[", 1)
        return GenericTypeRef(name.strip(), parse_type_ref(remainder[:-1].strip()))

    return SimpleTypeRef(text)


class TypeRegistry:
    _simple_types: dict[str, type[BaseModel]] = {}

    @classmethod
    def register_simple(cls, name: str, model_cls: type[BaseModel]) -> None:
        cls._simple_types[name] = model_cls

    @classmethod
    def get_model(cls, name: str) -> type[BaseModel]:
        if name not in cls._simple_types:
            raise KeyError(
                f"Unknown type '{name}'. Available: {sorted(cls._simple_types.keys())}"
            )
        return cls._simple_types[name]

    @classmethod
    def list_type_names(cls) -> list[str]:
        names = sorted(cls._simple_types.keys())
        names.extend(f"History[{name}]" for name in sorted(cls._simple_types.keys()))
        return names

    @classmethod
    def is_assignable(
        cls,
        source: str | PortTypeRef,
        target: str | PortTypeRef,
    ) -> bool:
        source_ref = parse_type_ref(source)
        target_ref = parse_type_ref(target)

        if isinstance(target_ref, UnionTypeRef):
            return any(cls.is_assignable(source_ref, option) for option in target_ref.options)
        if isinstance(source_ref, UnionTypeRef):
            return all(cls.is_assignable(option, target_ref) for option in source_ref.options)
        return source_ref == target_ref

    @classmethod
    def validate(
        cls,
        type_ref: str | PortTypeRef,
        value: Any,
    ) -> BaseModel:
        resolved = parse_type_ref(type_ref)

        if isinstance(resolved, UnionTypeRef):
            last_error: Exception | None = None
            for option in resolved.options:
                try:
                    return cls.validate(option, value)
                except Exception as exc:  # pragma: no cover - only used on failure
                    last_error = exc
            raise ValueError(
                f"Value did not match union type '{resolved}': {last_error}"
            )

        if isinstance(resolved, GenericTypeRef):
            if resolved.name != "History":
                raise ValueError(f"Unsupported generic type '{resolved.name}'.")

            history_model = cls.get_model("History")
            validated = history_model.model_validate(value)
            validated.items = [
                cls.validate(resolved.item_type, item) for item in validated.items
            ]
            return validated

        model_cls = cls.get_model(resolved.name)
        if isinstance(value, model_cls):
            return value
        return model_cls.model_validate(value)


def serialize_type_ref(type_ref: str | PortTypeRef) -> str:
    return str(parse_type_ref(type_ref))
