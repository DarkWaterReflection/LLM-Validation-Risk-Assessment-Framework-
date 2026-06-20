"""Render a Model Validation Report from a ``ValidationResult``.

Produces the 17-section MRM report as Markdown. HTML/PDF are optional follow-on
renders (markdown -> HTML via any renderer; HTML -> PDF via WeasyPrint extra).
The Markdown output is deterministic and diff-able, which matters for audit.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..schemas import ValidationResult

_TEMPLATE_DIR = Path(__file__).parent / "templates"


class ReportBuilder:
    def __init__(self, template_dir: Path | None = None) -> None:
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir or _TEMPLATE_DIR)),
            autoescape=select_autoescape(enabled_extensions=("html",)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_markdown(self, result: ValidationResult) -> str:
        template = self.env.get_template("validation_report.md.j2")
        return template.render(r=result)

    def write(self, result: ValidationResult, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        md = self.render_markdown(result)
        path = out_dir / f"validation_report_{result.run_id}.md"
        path.write_text(md, encoding="utf-8")
        return path
