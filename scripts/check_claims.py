"""Checks a PR against Station House's neuron claims.

Fails when the PR changes the explanation of a neuron someone else holds, or changes an explanation
without naming its neuron in the PR title or the "## Neuron(s)" section (that list is how Station
House links the PR to the claim), or without saying what the neuron does under "## In plain English".
Runs in CI on pull requests:

  PR_TITLE, PR_BODY, PR_AUTHOR_ID, BASE_SHA  set by .github/workflows/validate.yml
  CLAIMS_URL                                 optional, defaults to Station House's public claims map

Usage: python scripts/check_claims.py
"""

import json
import os
import re
import subprocess
import sys
import urllib.request

CLAIMS_URL = "https://app.station-house.net/api/public/marin-neuron-atlas/claims"
LAYERS = 64
NEURONS_PER_LAYER = 27648
ID_IN_TEXT = re.compile(r"\bL(\d{1,2})-N(\d{1,5})\b", re.IGNORECASE)
EXPLANATION_PATH = re.compile(r"^explanations/L\d{1,2}/(L\d{1,2}-N\d{1,5})\.ya?ml$")
SECTION = re.compile(r"^##\s*Neurons?(?:\(s\))?\s*$(.*?)(?=^##\s|\Z)", re.IGNORECASE | re.MULTILINE | re.DOTALL)
PLAIN_SECTION = re.compile(r"^##\s*In plain English\s*$(.*?)(?=^##\s|\Z)", re.IGNORECASE | re.MULTILINE | re.DOTALL)
PLAIN_MIN_WORDS = 10


def canonical(layer, neuron):
    layer, neuron = int(layer), int(neuron)
    if layer >= LAYERS or neuron >= NEURONS_PER_LAYER:
        return None
    return f"L{layer}-N{neuron}"


def ids_in(text):
    ids = []
    for match in ID_IN_TEXT.finditer(text):
        neuron_id = canonical(match.group(1), match.group(2))
        if neuron_id and neuron_id not in ids:
            ids.append(neuron_id)
    return ids


def neurons_in_pr(title, body):
    """Same rule as Station House: ids in the title and the "## Neuron(s)" section, ignoring HTML comments."""
    text = re.sub(r"<!--.*?-->", "", body or "", flags=re.DOTALL)
    match = SECTION.search(text)
    found = ids_in(title or "") + ids_in(match.group(1) if match else "")
    return list(dict.fromkeys(found))


def plain_english_problems(touched, body):
    """The error lines if the PR doesn't say, in its own words, what each touched neuron does."""
    match = PLAIN_SECTION.search(re.sub(r"<!--.*?-->", "", body or "", flags=re.DOTALL))
    text = match.group(1).strip() if match else ""
    if len(text.split()) < PLAIN_MIN_WORDS * len(touched):
        return [
            "Say what you believe the neuron does under '## In plain English' in the PR description: a sentence or "
            f"two per neuron, both sides (at least {PLAIN_MIN_WORDS} words each)."
        ]
    named = set(ids_in(text))
    if len(touched) > 1:
        return [
            f"{neuron_id}: name it under '## In plain English' with what you believe it does."
            for neuron_id in touched
            if neuron_id not in named
        ]
    return []


def explained_neuron(path):
    match = EXPLANATION_PATH.match(path)
    if not match:
        return None
    layer, neuron = match.group(1)[1:].split("-N")
    return canonical(layer, neuron)


def problems(touched, declared, claims, author_id):
    """The error lines for this PR, given the neurons whose explanations it changes."""
    errors = []
    for neuron_id in touched:
        if neuron_id not in declared:
            errors.append(
                f"{neuron_id}: this PR changes its explanation, so list it under '## Neuron(s)' in the PR "
                "description (or in the title)."
            )
        claim = claims.get(neuron_id) or {}
        holder = claim.get("holder")
        if holder and str(holder.get("githubId")) != str(author_id):
            if claim.get("state") == "in-review":
                held = f"whose PR #{claim.get('prNum')} for it is open"
            else:
                held = f"until {(claim.get('until') or '')[:10]}"
            errors.append(
                f"{neuron_id} is claimed by @{holder.get('github')}, {held}. Pick another neuron, or ask them to "
                "release it. Claims: https://app.station-house.net/projects/private/marin-neuron-atlas/neurons"
                f"#{neuron_id}"
            )
    return errors


def fetch_claims(url):
    with urllib.request.urlopen(url, timeout=20) as res:
        return json.load(res).get("claims", {})


def main():
    base = os.environ.get("BASE_SHA")
    if not base:
        print("check_claims: no BASE_SHA, so not a pull request; nothing to check.")
        return 0
    diff = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"], capture_output=True, text=True, check=True)
    touched = sorted({n for n in (explained_neuron(p) for p in diff.stdout.splitlines()) if n})
    if not touched:
        print("check_claims: no explanation files changed.")
        return 0
    declared = neurons_in_pr(os.environ.get("PR_TITLE"), os.environ.get("PR_BODY"))
    plain = plain_english_problems(touched, os.environ.get("PR_BODY"))
    try:
        claims = fetch_claims(os.environ.get("CLAIMS_URL") or CLAIMS_URL)
    except Exception as err:  # Station House down: don't block the PR on it.
        print(f"::warning::Couldn't read Station House's neuron claims ({err}); claim check skipped.")
        claims = {}
    errors = problems(touched, declared, claims, os.environ.get("PR_AUTHOR_ID")) + plain
    for message in errors:
        print(f"::error::{message}")
    for neuron_id in touched:
        if not (claims.get(neuron_id) or {}).get("holder"):
            print(
                f"::notice::{neuron_id} isn't claimed yet. This PR claims it for you once Station House sees it, "
                "if your GitHub account is verified on your Station House profile."
            )
    print(f"check_claims: {len(touched)} neuron(s): {'OK' if not errors else f'{len(errors)} problem(s)'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
