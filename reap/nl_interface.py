"""Natural-language descriptions for symbolic REAP programs."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

from .program import Op, Program

# Short human-facing summaries for known primitives. Unknown ops fall back to a
# generic description that still round-trips cleanly.
_OP_SUMMARIES: Dict[str, str] = {
    "rotate": "Rotate the grid",
    "flip": "Flip the grid",
    "map_color": "Remap colours",
    "keep_largest_object": "Keep the dominant object",
    "remove_color": "Remove specified colour",
    "mirror_symmetry": "Mirror grid to complete symmetry",
}


def _normalise_params(value: object) -> object:
    """Recursively convert JSON-parsed values into Python-native forms."""

    if isinstance(value, dict):
        normalised: Dict[object, object] = {}
        for key, val in value.items():
            new_key: object = key
            if isinstance(key, str) and key.lstrip("-").isdigit():
                new_key = int(key)
            normalised[new_key] = _normalise_params(val)
        return normalised
    if isinstance(value, list):
        return [_normalise_params(item) for item in value]
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return value


def _serialise_params(params: Dict[str, object]) -> str:
    """Return a stable JSON blob describing ``params``."""

    if not params:
        return ""
    return json.dumps(params, sort_keys=True)


def program_to_nl(program: Program) -> str:
    """Convert ``program`` into a bullet-point natural-language summary."""

    if not program.ops:
        return "- noop  # keep grid unchanged"
    lines: List[str] = []
    for op in program.ops:
        params_blob = _serialise_params(op.params)
        summary = _OP_SUMMARIES.get(op.name, f"Apply {op.name}")
        if params_blob:
            body = f"{op.name} {params_blob}"
        else:
            body = op.name
        lines.append(f"- {body}  # {summary}")
    return "\n".join(lines)


def nl_to_program(text: str) -> Program:
    """Parse ``text`` produced by :func:`program_to_nl` back into a program."""

    ops: List[Op] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        body = line[1:].strip()
        if not body:
            continue
        if "#" in body:
            body = body.split("#", 1)[0].strip()
        if not body or body == "noop":
            continue
        if " " in body:
            name, params_blob = body.split(" ", 1)
            params_blob = params_blob.strip()
        else:
            name, params_blob = body, ""
        params: Dict[str, object] = {}
        if params_blob:
            try:
                parsed = json.loads(params_blob)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid parameter JSON: {params_blob}") from exc
            params = _normalise_params(parsed)  # type: ignore[assignment]
        ops.append(Op(name, params))
    return Program(ops)


def roundtrip_program(program: Program) -> Program:
    """Return the program obtained after NL serialisation and parsing."""

    return nl_to_program(program_to_nl(program))


def programs_to_nl(programs: Iterable[Program]) -> List[str]:
    """Convenience helper for batch conversion."""

    return [program_to_nl(program) for program in programs]


__all__ = [
    "program_to_nl",
    "nl_to_program",
    "roundtrip_program",
    "programs_to_nl",
]
