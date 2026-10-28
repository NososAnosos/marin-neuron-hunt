"""Checks every request's [[ ]] targets the way the daily runner will: each must be exactly one Marin token,
prompts at most 512 tokens, named next tokens single tokens. Prints each target token so a valid-but-wrong
target (e.g. marking " and" instead of " bold") is easy to spot in the log.

Usage: python scripts/check_tokens.py
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.prompts import PromptError, check, encode, load_tokenizer, prompts_of  # noqa: E402


def main():
    tok = load_tokenizer()
    paths = sorted((ROOT / "requests").glob("*.yaml")) + sorted((ROOT / "examples").glob("L*-N*-[0-9][0-9][0-9].yaml"))
    errors = 0
    for path in paths:
        req = yaml.safe_load(path.read_text())
        print(f"{path.relative_to(ROOT)}")
        for text in prompts_of(req):
            try:
                print(f"  {encode(tok, text).token!r:18} {text}")
            except PromptError:
                pass
        for problem in check(tok, req):
            errors += 1
            print(f"  PROBLEM: {problem}")
    print(f"checked {len(paths)} requests: {'OK' if not errors else f'{errors} problem(s)'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
