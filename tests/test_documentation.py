"""Execute the Python code blocks embedded in Markdown documentation.

Documentation examples rot silently unless something runs them. The tutorial and
example scripts are already executed by CI; this module extends the same
guarantee to Markdown so that README and design-document snippets cannot drift
away from the implemented API.

Every ```python fence is executed. A fence that is deliberately illustrative
rather than runnable must be preceded by the marker comment
``<!-- docs-test: skip -->`` on its own line, and the marker must state why.
"""

from __future__ import annotations

import importlib
import re
import warnings
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SKIP_MARKER = "<!-- docs-test: skip"
_FENCE = re.compile(
    r"(?P<prefix>(?:^<!-- docs-test: skip[^\n]*-->\n)?)^```python\n(?P<body>.*?)^```",
    re.DOTALL | re.MULTILINE,
)


def _markdown_files() -> list[Path]:
    files = [REPOSITORY_ROOT / "README.md"]
    files.extend(sorted((REPOSITORY_ROOT / "docs").rglob("*.md")))
    return [path for path in files if path.is_file()]


def _blocks(path: Path) -> list[tuple[int, str, bool]]:
    text = path.read_text(encoding="utf-8")
    found: list[tuple[int, str, bool]] = []
    for match in _FENCE.finditer(text):
        line = text.count("\n", 0, match.start("body")) + 1
        found.append((line, match.group("body"), match.group("prefix").startswith(SKIP_MARKER)))
    return found


def _cases() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for path in _markdown_files():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        for line, body, skipped in _blocks(path):
            if skipped:
                continue
            cases.append((f"{relative}:{line}", body))
    return cases


CASES = _cases()


def test_markdown_python_blocks_were_discovered() -> None:
    """Guard against a regex change that silently stops collecting examples."""
    assert len(CASES) >= 20


_PUBLIC_SURFACE = re.compile(
    r"from chaos_numerics\.(?P<module>\w+) import \(\n(?P<names>.*?)\n\)",
    re.DOTALL,
)


@pytest.mark.parametrize(
    "module",
    ["core", "classical", "operators", "quantum", "spectral", "experiment"],
)
def test_public_api_document_lists_every_exported_name(module: str) -> None:
    """`public-api.md` sections 4.3 and 4.4 must enumerate the whole surface.

    Executing the block, which the test above already does, only catches a name in
    the document that the package does not export. The other direction is the one
    that drifts: a new public export is easy to add without touching the document,
    and a reader who treats the listing as the checklist it claims to be then never
    learns the name exists. Both `HusimiResult` and `WignerResult` went missing
    that way, along with every name added in this release.
    """
    text = (REPOSITORY_ROOT / "docs" / "design" / "public-api.md").read_text(encoding="utf-8")
    listed: set[str] = set()
    for match in _PUBLIC_SURFACE.finditer(text):
        if match.group("module") != module:
            continue
        listed |= {line.strip().rstrip(",") for line in match.group("names").splitlines()}
    assert listed, f"public-api.md has no import block for chaos_numerics.{module}"

    package = importlib.import_module(f"chaos_numerics.{module}")
    exported = set(package.__all__)
    assert not exported - listed, (
        f"chaos_numerics.{module} exports {sorted(exported - listed)} "
        "without listing them in docs/design/public-api.md"
    )


@pytest.mark.parametrize(("location", "source"), CASES, ids=[case[0] for case in CASES])
def test_markdown_python_block_executes(location: str, source: str) -> None:
    namespace: dict[str, object] = {"__name__": "__docs__"}
    # Library warnings are diagnostics that some examples deliberately provoke
    # (slow Lyapunov convergence, unlabelled symmetry sectors). They must not be
    # promoted to errors here, unlike in the rest of the suite.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        try:
            exec(compile(source, location, "exec"), namespace)
        except Exception as error:  # pragma: no cover - failure path is the point
            pytest.fail(
                f"documentation example at {location} failed with "
                f"{type(error).__name__}: {error}\n\n{source}"
            )
