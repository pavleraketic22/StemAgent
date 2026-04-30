from __future__ import annotations

import re
from typing import Any

from .base import BaseTool


class SecurityPatternScanTool(BaseTool):
    """Lightweight regex-based security scanner for code snippets/files."""

    name = "security_pattern_scan"
    description = "Scan code/text for common insecure patterns."

    _PATTERNS: dict[str, str] = {
        "hardcoded_secret": r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]+['\"]",
        "sql_injection_risk": r"(?i)(select|insert|update|delete).*(\+|f['\"]|format\()",
        "command_injection_risk": r"(?i)(os\.system|subprocess\.(Popen|run)\()",
        "unsafe_deserialization": r"(?i)(pickle\.loads|yaml\.load\()",
        "weak_hash": r"(?i)(md5\(|sha1\()",
    }

    def run(self, context: dict[str, Any]) -> str:
        text = str(context.get("text") or context.get("code") or "")
        if not text and context.get("file_path"):
            try:
                from pathlib import Path

                text = Path(str(context["file_path"])).read_text(encoding="utf-8")
            except Exception as exc:
                return f"security_pattern_scan file read error: {exc}"

        if not text:
            return "security_pattern_scan: provide 'text'/'code' or 'file_path'."

        findings: list[str] = []
        for name, pattern in self._PATTERNS.items():
            for match in re.finditer(pattern, text, flags=re.MULTILINE):
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 40)
                evidence = " ".join(text[start:end].split())
                findings.append(f"[{name}] {evidence}")

        if not findings:
            return "security_pattern_scan: no obvious pattern hits."

        max_findings = int(context.get("security_max_findings", 20))
        return "\n".join(findings[:max_findings])
