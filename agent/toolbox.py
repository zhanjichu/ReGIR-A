"""Run configured restoration tools; model output never becomes a command."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .prompts import TOOLS


def _expand(template: str, values: dict[str, str]) -> str:
    pattern = r"\{(" + "|".join(re.escape(key) for key in values) + r")\}"
    return re.sub(pattern, lambda match: values[match.group(1)], template)


class ToolExecutor:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        with self.config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream) if self.config_path.suffix.lower() == ".json" else yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError("The tool configuration must be a mapping.")
        self.legacy_json = "tools" not in config
        self.tools = config if self.legacy_json else config["tools"]
        if not isinstance(self.tools, dict) or not self.tools:
            raise ValueError("Expected a non-empty tools mapping.")
        unknown = set(self.tools) - set(TOOLS)
        if unknown:
            raise ValueError(f"Unknown tool names: {sorted(unknown)}")

    def validate(self) -> None:
        """Check recipes before loading a multi-billion-parameter model."""
        missing = set(TOOLS) - set(self.tools)
        if missing:
            raise ValueError(f"Missing tool configurations: {sorted(missing)}")
        for name, config in self.tools.items():
            if not isinstance(config, dict) or not (config.get("command") or config.get("commands")):
                raise ValueError(f"No command configured for {name}.")
            self._working_dir(config, name)

    def _working_dir(self, config: dict, name: str) -> Path | None:
        cwd = config.get("working_dir", config.get("path"))
        if not cwd:
            return None
        directory = Path(cwd).expanduser()
        if not directory.is_absolute():
            directory = self.config_path.parent / directory
        if not directory.is_dir():
            raise FileNotFoundError(f"Working directory for {name} does not exist: {directory}")
        return directory.resolve()

    def apply(self, tool_name: str, image_path: str | Path, workspace: str | Path, step: int) -> Path:
        if tool_name not in self.tools:
            raise ValueError(f"Unknown restoration action: {tool_name}")
        config: dict[str, Any] = self.tools[tool_name]
        source = Path(image_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        step_root = Path(workspace).resolve() / f"step_{step:04d}_{tool_name}"
        input_dir, output_dir = step_root / "input", step_root / "output"
        input_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        staged_input = input_dir / source.name
        shutil.copy2(source, staged_input)
        cwd = self._working_dir(config, tool_name)
        values = {
            "input_dir": str(input_dir), "output_dir": str(output_dir),
            "input": str(input_dir), "output": str(output_dir),
            "input_file": str(staged_input), "filename": source.name,
            "stem": source.stem, "tool_dir": str(cwd or self.config_path.parent),
        }
        for key, value in config.get("variables", {}).items():
            values[key] = _expand(str(value), values)

        command = config.get("command")
        if self.legacy_json:
            if not isinstance(command, str) or not command.strip():
                raise ValueError(f"Original JSON recipe for {tool_name} needs a command string.")
            quoted = {key: shlex.quote(value) for key, value in values.items()}
            commands = [["bash", "-e", "-c", _expand(command, quoted)]]
        else:
            recipes = config.get("commands")
            if recipes is None:
                recipes = [command]
            if not isinstance(recipes, list) or not recipes:
                raise ValueError(f"No command configured for {tool_name}.")
            commands = []
            for recipe in recipes:
                if isinstance(recipe, str):
                    if "\n" in recipe:
                        raise ValueError("Use a commands list for multiline YAML recipes.")
                    parts = shlex.split(recipe)
                elif isinstance(recipe, list):
                    parts = [str(part) for part in recipe]
                else:
                    raise ValueError("Each command must be an argument list or single-line string.")
                if not parts:
                    raise ValueError(f"Empty command for {tool_name}.")
                commands.append([_expand(part, values) for part in parts])

        for command_parts in commands:
            try:
                result = subprocess.run(
                    command_parts, cwd=cwd, check=False, capture_output=True,
                    text=True, timeout=int(config.get("timeout_seconds", 3600)),
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Restoration command timed out for {tool_name}.") from exc
            if result.returncode:
                details = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(f"{tool_name} failed ({result.returncode}): {details[-2000:]}")

        if config.get("output_file"):
            restored = Path(_expand(config["output_file"], values))
            if not restored.is_absolute():
                restored = output_dir / restored
        else:
            restored = output_dir / source.name
            if not restored.is_file():
                matches = list(output_dir.rglob(source.name))
                if len(matches) == 1:
                    restored = matches[0]
        if not restored.is_file():
            raise FileNotFoundError(f"{tool_name} did not create its expected output: {restored}")
        from PIL import Image

        with Image.open(restored) as image:
            image.verify()
        return restored
