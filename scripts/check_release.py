"""Check tracked release files, including decompressed data, without printing secrets.

This narrow check complements a full Gitleaks scan; it is not a credential detector
for every provider. It also prevents accidentally committing the manuscript.
"""

from pathlib import Path
import gzip
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private-key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "openai-key": rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}",
    "github-token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "aws-access-id": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "google-api-key": rb"\bAIza[A-Za-z0-9_-]{35}\b",
    "slack-token": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}",
}
FORBIDDEN = {".tex", ".bib", ".pdf", ".pem", ".key", ".aux", ".bbl", ".blg"}


def main():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    problems = []
    for name in filter(None, names):
        path = ROOT / name
        if path.suffix.lower() in FORBIDDEN or any(
            part.lower() in {"paper", "overleaf"} for part in path.parts
        ):
            problems.append((name, "manuscript-or-private-file"))
        if path.name.startswith(".env") and path.name != ".env.example":
            problems.append((name, "environment-file"))
        payload = path.read_bytes()
        if path.suffix == ".gz":
            payload = gzip.decompress(payload)
        for kind, pattern in PATTERNS.items():
            if re.search(pattern, payload):
                problems.append((name, kind))
    for path, kind in problems:
        print(f"{kind}: {path} [content redacted]")
    if problems:
        raise SystemExit(1)
    print(
        f"Checked {sum(bool(n) for n in names)} tracked files; no prohibited files or recognized credentials."
    )


if __name__ == "__main__":
    main()
