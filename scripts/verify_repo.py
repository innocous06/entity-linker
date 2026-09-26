"""Pre-publish verification script checking Appendix C compliance."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BANNED_WORDS = [
    "production-grade", "enterprise-grade", "industry-standard", "zero bugs",
    "100% pass", "secure", "safe", "blazing fast", "high-performance",
    "complete implementation", "full implementation"
]

AI_TERMS = [
    "antigravity", "gemini", "chatgpt", "openai", "claude",
    "deepseek", "copilot", "made with"
]

EXCLUDED_PARTS = {"cache", "workspace", "__pycache__", ".git", "pipeline_v2.ipynb", "pipeline.py", "pipeline_v2.py", "verify_repo.py"}

def main():
    files = [
        p for p in ROOT.rglob("*")
        if p.is_file()
        and not any(x in p.parts for x in EXCLUDED_PARTS)
        and p.suffix in {".py", ".md", ".txt", ".tsv"}
    ]

    print(f"Auditing {len(files)} files in repository...")
    issues = []

    for path in files:
        content = path.read_text(encoding="utf-8", errors="ignore")

        for bw in BANNED_WORDS:
            if bw in content.lower():
                issues.append(f"{path.relative_to(ROOT)}: Banned claim word '{bw}'")

        for ai in AI_TERMS:
            if ai in content.lower():
                issues.append(f"{path.relative_to(ROOT)}: AI attribution term '{ai}'")

        emojis = re.findall(r"[\U00010000-\U0010ffff]", content)
        if emojis:
            issues.append(f"{path.relative_to(ROOT)}: Emoji detected: {emojis[:3]}")

        # Check for smart quotes or dashes
        smart_chars = [c for c in content if c in "“”‘’—–"]
        if smart_chars:
            issues.append(f"{path.relative_to(ROOT)}: Smart quote or en/em dash detected: {smart_chars[:3]}")

    if issues:
        print("[FAIL] Issues detected:")
        for iss in issues:
            print(f"  - {iss}")
        sys.exit(1)
    else:
        print("[OK] All checks passed: 0 banned words, 0 emojis, 0 AI attribution, 0 smart quotes.")

if __name__ == "__main__":
    main()
