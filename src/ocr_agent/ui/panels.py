"""Panel components for OCR Agent stage display."""

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ocr_agent.ui.theme import (
    ENGINE_ICONS,
    ENGINE_LABELS,
    STAGE_COLORS,
    STATUS_ICONS,
)


class StagePanel:
    """A panel representing a processing stage."""

    def __init__(
        self,
        stage_num: int,
        title: str,
        subtitle: str = "",
        color: str | None = None,
    ):
        self.stage_num = stage_num
        self.title = title
        self.subtitle = subtitle
        self.color = color or STAGE_COLORS.get(title.lower(), "bright_blue")
        self.content_lines: list[RenderableType] = []

    def add_engine_header(self, engine: str, description: str = "") -> None:
        """Add an engine header line."""
        icon = ENGINE_ICONS.get(engine, "⚙")
        label = ENGINE_LABELS.get(engine, engine)

        line = Text()
        line.append(f"{icon} ", style="bold")
        line.append(label, style=engine)
        if description:
            line.append(f" {description}", style="dim")

        self.content_lines.append(line)
        self.content_lines.append(Text())  # Spacing

    def add_progress_line(
        self,
        current: int,
        total: int,
        label: str = "",
        width: int = 40,
    ) -> None:
        """Add a progress bar line."""
        pct = current / max(total, 1)
        filled = int(pct * width)
        bar = "█" * filled + "░" * (width - filled)

        line = Text()
        line.append(bar, style="bright_blue")
        line.append(f" {current}/{total}", style="dim")
        if label:
            line.append(f" {label}", style="dim")

        self.content_lines.append(line)

    def add_result(
        self,
        item: str,
        status: str,
        message: str = "",
        confidence: float | None = None,
    ) -> None:
        """Add a result line."""
        icon = STATUS_ICONS.get(status, "?")

        line = Text()
        line.append(f"{icon} ", style=status)
        line.append(item, style="bold" if status != "success" else "")

        if confidence is not None:
            conf_style = "success" if confidence >= 0.8 else "warning" if confidence >= 0.6 else "error"
            line.append(f" ({confidence:.0%})", style=conf_style)

        if message:
            line.append(f" - {message}", style="dim")

        self.content_lines.append(line)

    def add_metric(self, label: str, value: str, status: str = "info") -> None:
        """Add a metric line."""
        line = Text()
        line.append(f"   {label}: ", style="dim")
        line.append(value, style=status)
        self.content_lines.append(line)

    def add_cost(self, amount: float, description: str = "") -> None:
        """Add a cost line."""
        line = Text()
        line.append("💰 ", style="bold")
        line.append("Cost: ", style="dim")
        line.append(f"${amount:.4f}", style="warning")
        if description:
            line.append(f" ({description})", style="dim")
        self.content_lines.append(line)

    def add_text(self, text: str | Text) -> None:
        """Add arbitrary text."""
        if isinstance(text, str):
            text = Text(text)
        self.content_lines.append(text)

    def add_spacing(self) -> None:
        """Add a blank line."""
        self.content_lines.append(Text())

    def render(self) -> Panel:
        """Render the panel."""
        title = Text()
        title.append(f"STAGE {self.stage_num}: ", style=f"bold {self.color}")
        title.append(self.title, style="bold white")

        return Panel(
            Group(*self.content_lines),
            title=title,
            subtitle=self.subtitle if self.subtitle else None,
            border_style=self.color,
            padding=(1, 2),
        )


class SummaryPanel:
    """Final summary panel showing processing results."""

    def __init__(self):
        self.pages_success = 0
        self.pages_total = 0
        self.figures_count = 0
        self.time_seconds = 0.0
        self.cost = 0.0
        self.engines_used: dict[str, int] = {}
        self.output_path = ""
        self.output_files: list[str] = []

    def set_stats(
        self,
        pages_success: int,
        pages_total: int,
        figures_count: int = 0,
        time_seconds: float = 0.0,
        cost: float = 0.0,
    ) -> None:
        """Set processing statistics."""
        self.pages_success = pages_success
        self.pages_total = pages_total
        self.figures_count = figures_count
        self.time_seconds = time_seconds
        self.cost = cost

    def add_engine_usage(self, engine: str, count: int) -> None:
        """Record engine usage."""
        self.engines_used[engine] = self.engines_used.get(engine, 0) + count

    def set_output(self, path: str, files: list[str] | None = None) -> None:
        """Set output location."""
        self.output_path = path
        self.output_files = files or []

    def render(self) -> Panel:
        """Render the summary panel."""
        sections = []

        # Stats section
        stats = Text()
        stats.append("📊 ", style="bold")
        stats.append("Summary\n", style="bold white")
        stats.append(f"   Pages: {self.pages_success}/{self.pages_total} successful\n")

        if self.figures_count > 0:
            stats.append(f"   Figures: {self.figures_count} described\n")

        stats.append(f"   Time: {self.time_seconds:.1f}s\n")

        if self.cost > 0:
            stats.append(f"   Cost: ${self.cost:.4f}\n", style="warning")

        sections.append(stats)

        # Engines section
        if self.engines_used:
            engines = Text()
            engines.append("\n🔧 ", style="bold")
            engines.append("Engines Used\n", style="bold white")

            items = list(self.engines_used.items())
            for i, (engine, count) in enumerate(items):
                icon = ENGINE_ICONS.get(engine, "⚙")
                label = ENGINE_LABELS.get(engine, engine)
                prefix = "└──" if i == len(items) - 1 else "├──"

                engines.append(f"   {prefix} {icon} ", style="dim")
                engines.append(label, style=engine)
                engines.append(f": {count}\n")

            sections.append(engines)

        # Output section
        if self.output_path:
            output = Text()
            output.append("\n📁 ", style="bold")
            output.append("Output: ", style="bold white")
            output.append(self.output_path, style="info")

            if self.output_files:
                for i, f in enumerate(self.output_files):
                    prefix = "└──" if i == len(self.output_files) - 1 else "├──"
                    output.append(f"\n   {prefix} {f}", style="dim")

            sections.append(output)

        return Panel(
            Group(*sections),
            title="[bold green]✨ COMPLETE[/bold green]",
            border_style="green",
            padding=(1, 2),
        )


class AuditPanel:
    """Panel for displaying audit results."""

    def __init__(self):
        self.metrics: list[dict] = []
        self.llm_results: list[dict] = []

    def add_metric(
        self,
        name: str,
        value: str,
        threshold: str | None = None,
        passed: bool = True,
    ) -> None:
        """Add an audit metric."""
        self.metrics.append({
            "name": name,
            "value": value,
            "threshold": threshold,
            "passed": passed,
        })

    def add_llm_review(
        self,
        item: str,
        verdict: str,
        reason: str = "",
    ) -> None:
        """Add LLM review result."""
        self.llm_results.append({
            "item": item,
            "verdict": verdict,
            "reason": reason,
        })

    def render(self) -> Panel:
        """Render the audit panel."""
        sections = []

        # Heuristics section
        if self.metrics:
            heuristics = Text()
            heuristics.append("📊 ", style="bold")
            heuristics.append("Heuristics Check\n", style="bold white")

            for m in self.metrics:
                icon = STATUS_ICONS["success"] if m["passed"] else STATUS_ICONS["warning"]
                style = "success" if m["passed"] else "warning"

                heuristics.append(f"   {icon} ", style=style)
                heuristics.append(f"{m['name']}: ", style="dim")
                heuristics.append(m["value"], style=style)

                if m["threshold"]:
                    heuristics.append(f" (threshold: {m['threshold']})", style="dim")

                heuristics.append("\n")

            sections.append(heuristics)

        # LLM section
        if self.llm_results:
            llm = Text()
            llm.append("\n🤖 ", style="bold")
            llm.append("LLM Review\n", style="bold white")

            for r in self.llm_results:
                status = "success" if r["verdict"] == "acceptable" else "warning"
                icon = STATUS_ICONS["success"] if r["verdict"] == "acceptable" else STATUS_ICONS["warning"]

                llm.append(f"   {icon} ", style=status)
                llm.append(f"{r['item']}: ", style="bold")
                llm.append(r["verdict"], style=status)

                if r["reason"]:
                    llm.append(f" - {r['reason']}", style="dim")

                llm.append("\n")

            sections.append(llm)

        return Panel(
            Group(*sections),
            title="[bold green]Quality Audit[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
