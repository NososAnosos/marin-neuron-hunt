"""Notes (never fails) a pull request that adds or changes an experiment request without a
`Design-Session:` line in its description, i.e. without a Station House Experimentation Wizard review.

  python scripts/design_session_note.py    in CI: BASE_SHA and PR_BODY, set by validate.yml

The wizard is three rounds of reviewer questions on the design that the author answers (see
CONTRIBUTING.md, "Before you open the PR"). This step only warns: the PR gets a GitHub warning and a
line in the run summary. Whether the session is complete and matches the design is checked by the
Station House watcher, which records it on the platform.
"""
import os
import re
import subprocess
import sys

REQUEST = re.compile(r"^requests/[^/]+(?:\.ya?ml|/DESIGN\.md)$")
SESSION = re.compile(r"Design-Session:\s*(ds_[A-Za-z0-9]{22})\b")

WARNING = ("This PR adds or changes an experiment request, but its description has no `Design-Session:` line, so the "
           "design hasn't been through the Station House Experimentation Wizard. Run it from your coding agent (Claude "
           "Code: the stationhouse plugin; others: `stationhouse design start` / `design round`), then add the line it "
           "gives you to the description. This doesn't block merging.")


def note(paths, body):
    """(level, message) for this PR: ('none', ...) when it has no request, else 'ok' or 'warning'."""
    if not any(REQUEST.match(p) for p in paths):
        return "none", "No experiment request in this PR; no design review needed."
    m = SESSION.search(body or "")
    if m:
        return "ok", f"Design review {m.group(1)} is linked; the Station House watcher checks it's complete."
    return "warning", WARNING


def main():
    base = os.environ.get("BASE_SHA")
    if not base:
        print("design_session_note: not a pull request; nothing to check.")
        return 0
    diff = subprocess.run(["git", "diff", "--name-only", "--diff-filter=AM", f"{base}...HEAD"],
                          capture_output=True, text=True, check=False)
    level, message = note(diff.stdout.splitlines(), os.environ.get("PR_BODY") or "")
    print(f"::warning title=Design review::{message}" if level == "warning" else message)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and level != "none":
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"### Design review\n\n{'⚠️ ' if level == 'warning' else '✅ '}{message}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
