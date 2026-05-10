from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .settings import OpsSettings


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        chunks.append(text)
            elif isinstance(content, str):
                chunks.append(content)
        if chunks:
            return "\n".join(chunks)
    return str(response)


def _resolve_backend(settings: OpsSettings) -> str:
    backend = settings.codex_backend.strip().lower()
    if backend not in {"auto", "api", "cli"}:
        raise RuntimeError("GA_LAB_CODEX_BACKEND must be one of: auto, api, cli")
    if backend == "auto":
        return "api" if os.getenv("OPENAI_API_KEY") else "cli"
    return backend


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _invoke_codex_api(
    *,
    settings: OpsSettings,
    prompt: str,
    model: str | None,
    metadata: dict[str, str] | None,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Codex integration requires the optional 'openai' dependency") from exc

    resolved_model = model or settings.codex_model
    if not resolved_model:
        raise RuntimeError("Set GA_LAB_CODEX_MODEL or pass a model explicitly")

    client = OpenAI()
    response = client.responses.create(
        model=resolved_model,
        input=prompt,
        metadata=metadata or {},
    )
    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", resolved_model),
        "output_text": _response_text(response),
        "backend": "api",
    }


def _ensure_cli_available(command_name: str, *, cwd: Path) -> str:
    executable = shutil.which(command_name)
    if executable is None:
        raise RuntimeError(
            f"Codex CLI command '{command_name}' was not found. Install Codex CLI or set "
            "GA_LAB_CODEX_CLI_COMMAND."
        )
    status = _run_command([executable, "login", "status"], cwd=cwd)
    if status.returncode != 0:
        message = status.stderr.strip() or status.stdout.strip()
        raise RuntimeError(
            "Codex CLI is not logged in. Run `codex login` first."
            + (f" Details: {message}" if message else "")
        )
    return executable


def _invoke_codex_cli(
    *,
    settings: OpsSettings,
    prompt: str,
    model: str | None,
    metadata: dict[str, str] | None,
) -> dict[str, Any]:
    executable = _ensure_cli_available(settings.codex_cli_command, cwd=settings.project_root)
    resolved_model = model or settings.codex_model
    prompt_text = prompt
    if metadata:
        prompt_text = (
            "Context metadata:\n"
            f"{metadata}\n\n"
            "Task:\n"
            f"{prompt}"
        )

    with tempfile.NamedTemporaryFile(
        mode="w+",
        prefix="ga_codex_cli_",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as handle:
        output_path = Path(handle.name)

    try:
        command = [
            executable,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--cd",
            str(settings.project_root),
            "--color",
            "never",
            "--ephemeral",
            "--output-last-message",
            str(output_path),
        ]
        if resolved_model:
            command.extend(["--model", resolved_model])
        command.append("-")

        completed = _run_command(command, cwd=settings.project_root, input_text=prompt_text)
        output_text = (
            output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip() or output_text
            raise RuntimeError(details or "Codex CLI execution failed")
        if not output_text:
            output_text = completed.stdout.strip()
        if not output_text:
            raise RuntimeError("Codex CLI completed without producing a final message")
        return {
            "id": None,
            "model": resolved_model or "cli-default",
            "output_text": output_text,
            "backend": "cli",
        }
    finally:
        output_path.unlink(missing_ok=True)


def invoke_codex(
    *,
    settings: OpsSettings,
    prompt: str,
    model: str | None = None,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    backend = _resolve_backend(settings)
    if backend == "api":
        return _invoke_codex_api(
            settings=settings,
            prompt=prompt,
            model=model,
            metadata=metadata,
        )
    return _invoke_codex_cli(
        settings=settings,
        prompt=prompt,
        model=model,
        metadata=metadata,
    )
