"""Main console interface for OCR Agent."""

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ocr_agent import __version__
from ocr_agent.ui.theme import (
    AGENT_THEME,
    ENGINE_ICONS,
    ENGINE_LABELS,
    STATUS_ICONS,
)


class AgentConsole:
    """Beautiful terminal interface for the OCR Agent."""

    def __init__(self, verbose: bool = False):
        self.console = Console(theme=AGENT_THEME)
        self.verbose = verbose

    def print_header(self) -> None:
        """Print the agent header banner."""
        header = Text()
        header.append("🔍 ", style="bold")
        header.append("OCR Agent", style="bold white")
        header.append(f" v{__version__}", style="dim")

        subtitle = Text("Multi-Engine Document Processing", style="dim italic")

        panel = Panel(
            Group(header, subtitle),
            border_style="bright_blue",
            padding=(0, 2),
        )
        self.console.print(panel)
        self.console.print()

    def print_document_info(
        self,
        filename: str,
        pages: int,
        size_mb: float,
        doc_type: str | None = None,
        detected_features: list[str] | None = None,
    ) -> None:
        """Print document information."""
        info = Text()
        info.append("📄 ", style="bold")
        info.append("Document: ", style="dim")
        info.append(filename, style="bold white")
        info.append(f" ({pages} pages, {size_mb:.1f} MB)", style="dim")

        self.console.print(info)

        if doc_type or detected_features:
            details = Text("   ")
            details.append("Type: ", style="dim")
            details.append(doc_type or "Unknown", style="info")

            if detected_features:
                details.append(" (detected: ", style="dim")
                details.append(", ".join(detected_features), style="warning")
                details.append(")", style="dim")

            self.console.print(details)

        self.console.print()

    def print_stage_header(self, stage_num: int, title: str, subtitle: str = "") -> None:
        """Print a stage header."""
        header = Text()
        header.append(f"STAGE {stage_num}: ", style="bold bright_blue")
        header.append(title, style="bold white")

        self.console.print()
        self.console.print(Panel(
            header,
            subtitle=subtitle if subtitle else None,
            border_style="bright_blue",
            padding=(0, 1),
        ))

    def print_engine_active(self, engine: str, description: str = "") -> None:
        """Print which engine is currently active."""
        icon = ENGINE_ICONS.get(engine, "⚙")
        label = ENGINE_LABELS.get(engine, engine)

        line = Text()
        line.append(f"{icon} ", style="bold")
        line.append(label, style=engine)
        if description:
            line.append(f" {description}", style="dim")

        self.console.print(line)

    def print_page_result(
        self,
        page: int,
        status: str,
        message: str = "",
        confidence: float | None = None,
    ) -> None:
        """Print result for a single page."""
        icon = STATUS_ICONS.get(status, "?")
        style = status

        line = Text("   ")
        line.append(f"{icon} ", style=style)
        line.append(f"Page {page}", style="bold" if status != "success" else "")

        if confidence is not None:
            conf_style = "success" if confidence >= 0.8 else "warning" if confidence >= 0.6 else "error"
            line.append(f" ({confidence:.0%})", style=conf_style)

        if message:
            line.append(f" - {message}", style="dim")

        self.console.print(line)

    def print_audit_result(
        self,
        metric: str,
        value: str,
        status: str = "info",
    ) -> None:
        """Print an audit metric result."""
        icon = STATUS_ICONS.get(status, "○")

        line = Text("   ")
        line.append(f"{icon} ", style=status)
        line.append(f"{metric}: ", style="dim")
        line.append(value, style=status)

        self.console.print(line)

    def print_cost(self, amount: float, description: str = "") -> None:
        """Print cost information."""
        line = Text("   ")
        line.append("💰 ", style="bold")
        line.append("Cost: ", style="dim")
        line.append(f"${amount:.4f}", style="warning")
        if description:
            line.append(f" ({description})", style="dim")

        self.console.print(line)

    def print_figure_result(
        self,
        figure_num: int,
        page: int,
        fig_type: str,
        description: str,
    ) -> None:
        """Print result for a processed figure."""
        line = Text("   ")
        line.append(f"{STATUS_ICONS['success']} ", style="success")
        line.append(f"Figure {figure_num}", style="bold")
        line.append(f" (p.{page}): ", style="dim")
        line.append(fig_type, style="info")
        line.append(f" - {description[:50]}{'...' if len(description) > 50 else ''}", style="dim")

        self.console.print(line)

    def print_summary(
        self,
        pages_success: int,
        pages_total: int,
        figures_count: int,
        time_seconds: float,
        cost: float,
        engines_used: dict[str, int],
        output_path: str,
    ) -> None:
        """Print the final summary panel."""
        self.console.print()

        # Build summary content
        content = []

        # Stats section
        stats = Text()
        stats.append("📊 ", style="bold")
        stats.append("Summary\n", style="bold white")
        stats.append(f"   Pages: {pages_success}/{pages_total} successful\n", style="dim")
        stats.append(f"   Figures: {figures_count} described\n", style="dim")
        stats.append(f"   Time: {time_seconds:.1f}s\n", style="dim")
        stats.append(f"   Cost: ${cost:.4f}\n", style="dim")
        content.append(stats)

        # Engines section
        engines = Text()
        engines.append("\n🔧 ", style="bold")
        engines.append("Engines Used\n", style="bold white")

        engine_items = list(engines_used.items())
        for i, (engine, count) in enumerate(engine_items):
            icon = ENGINE_ICONS.get(engine, "⚙")
            label = ENGINE_LABELS.get(engine, engine)
            prefix = "└──" if i == len(engine_items) - 1 else "├──"
            engines.append(f"   {prefix} {icon} ", style="dim")
            engines.append(label, style=engine)
            engines.append(f": {count}\n", style="dim")

        content.append(engines)

        # Output section
        output = Text()
        output.append("\n📁 ", style="bold")
        output.append("Output: ", style="bold white")
        output.append(output_path, style="info")
        content.append(output)

        panel = Panel(
            Group(*content),
            title="[bold green]✨ COMPLETE[/bold green]",
            border_style="green",
            padding=(1, 2),
        )

        self.console.print(panel)

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[error]{STATUS_ICONS['error']} Error:[/error] {message}")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[warning]{STATUS_ICONS['warning']} Warning:[/warning] {message}")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        if self.verbose:
            self.console.print(f"[dim]ℹ {message}[/dim]")

    def rule(self, title: str = "") -> None:
        """Print a horizontal rule."""
        self.console.print(Rule(title, style="dim"))
