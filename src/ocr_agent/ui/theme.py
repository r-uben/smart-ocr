"""Color theme and styling for OCR Agent terminal UI."""

from rich.style import Style
from rich.theme import Theme

# Agent colors
NOUGAT_COLOR = "#E67E22"      # Orange - scientific specialist
DEEPSEEK_COLOR = "#3498DB"    # Blue - local general
MISTRAL_COLOR = "#9B59B6"     # Purple - cloud fallback
GEMINI_COLOR = "#1ABC9C"      # Teal - cloud fallback
OLLAMA_COLOR = "#2ECC71"      # Green - local audit

# Status colors
SUCCESS_COLOR = "#2ECC71"
WARNING_COLOR = "#F39C12"
ERROR_COLOR = "#E74C3C"
INFO_COLOR = "#3498DB"
DIM_COLOR = "#7F8C8D"

# Stage colors
STAGE_COLORS = {
    "classify": "#9B59B6",
    "primary": "#3498DB",
    "audit": "#2ECC71",
    "fallback": "#E67E22",
    "figures": "#1ABC9C",
}

# Engine styling
ENGINE_STYLES = {
    "nougat": Style(color=NOUGAT_COLOR, bold=True),
    "deepseek": Style(color=DEEPSEEK_COLOR, bold=True),
    "mistral": Style(color=MISTRAL_COLOR, bold=True),
    "gemini": Style(color=GEMINI_COLOR, bold=True),
    "ollama": Style(color=OLLAMA_COLOR, bold=True),
}

ENGINE_ICONS = {
    "nougat": "🥜",
    "deepseek": "🔷",
    "mistral": "💎",
    "gemini": "✨",
    "ollama": "🦙",
}

ENGINE_LABELS = {
    "nougat": "Nougat",
    "deepseek": "DeepSeek",
    "mistral": "Mistral",
    "gemini": "Gemini",
    "ollama": "Ollama",
}

STATUS_ICONS = {
    "success": "✓",
    "warning": "⚠",
    "error": "✗",
    "pending": "○",
    "running": "◉",
    "skipped": "◌",
}

# Create Rich theme
AGENT_THEME = Theme({
    "nougat": ENGINE_STYLES["nougat"],
    "deepseek": ENGINE_STYLES["deepseek"],
    "mistral": ENGINE_STYLES["mistral"],
    "gemini": ENGINE_STYLES["gemini"],
    "ollama": ENGINE_STYLES["ollama"],
    "success": Style(color=SUCCESS_COLOR),
    "warning": Style(color=WARNING_COLOR),
    "error": Style(color=ERROR_COLOR),
    "info": Style(color=INFO_COLOR),
    "dim": Style(color=DIM_COLOR),
    "stage.title": Style(color="#ECF0F1", bold=True),
    "stage.border": Style(color="#34495E"),
    "header": Style(color="#ECF0F1", bold=True),
    "highlight": Style(color="#F1C40F", bold=True),
})
