"""Opens one results PR per request and links it from the request's PR (EXPERIMENTS.md, "Results").

Each results PR is branched from the same base commit, titled `Results <request id>: <outcome>`, and commits
only `results/<request id>.yaml`. The PR that added the request, and the explanation PR for the neuron when
that's a different one, each get a comment linking the results PR.
"""

import datetime
import json
import os
import subprocess

from runner import report
from runner.pick import ROOT

REPO = os.environ.get("GITHUB_REPOSITORY", "Station-House/marin-neuron-hunt")


def sh(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, cwd=ROOT).stdout


def neuron_of(request_id):
    return request_id.rsplit("-", 1)[0]


def explanation_path(request_id):
    neuron = neuron_of(request_id)
    return f"explanations/{neuron.split('-')[0]}/{neuron}.yaml"


def title(rec):
    return f"Results {rec['request']}: {rec['outcome']}"


def branch_name(request_id, remote_branches, now):
    """results-<id>, or results-<id>-<UTC stamp> when that branch already exists (a rerun)."""
    name = f"results-{request_id}"
    if name in remote_branches:
        name += now.strftime("-%Y%m%d-%H%M")
    return name


def remote_branches(run=sh):
    out = run("git", "ls-remote", "--heads", "origin", "results-*")
    return {line.split("refs/heads/", 1)[1] for line in out.splitlines() if "refs/heads/" in line}


def all_prs(run=sh):
    return json.loads(run("gh", "pr", "list", "-R", REPO, "--state", "all", "--limit", "300",
                          "--json", "number,state,files,mergedAt,headRefName"))


def request_pr(commit, prs, request_id, run=sh):
    """The PR that added the request: the PR containing the commit that added the file, else any merged PR
    touching the file."""
    if commit:
        try:
            found = json.loads(run("gh", "api", f"repos/{REPO}/commits/{commit}/pulls"))
        except subprocess.CalledProcessError:
            found = []
        merged = [p for p in found if p.get("merged_at")]
        if merged or found:
            return (merged or found)[0]["number"]
    path = f"requests/{request_id}.yaml"
    touching = [p for p in prs if p["state"] == "MERGED" and any(f["path"] == path for f in p.get("files") or [])]
    return max(touching, key=lambda p: p["mergedAt"] or "")["number"] if touching else None


def explanation_pr(request_id, prs):
    """The neuron's explanation PR: an open one if there is one, else the most recently merged."""
    path = explanation_path(request_id)
    touching = [p for p in prs if not p["headRefName"].startswith("results-")
                and any(f["path"] == path for f in p.get("files") or [])]
    open_ = [p for p in touching if p["state"] == "OPEN"]
    if open_:
        return max(open_, key=lambda p: p["number"])["number"]
    merged = [p for p in touching if p["state"] == "MERGED"]
    return max(merged, key=lambda p: p["mergedAt"] or "")["number"] if merged else None


def link_targets(rec, prs, run=sh):
    out = []
    for n in (request_pr(rec["request_commit"], prs, rec["request"], run), explanation_pr(rec["request"], prs)):
        if n and n not in out:
            out.append(n)
    return out


def pr_body(rec, targets):
    linked = ", ".join(f"#{n}" for n in targets) or "none found"
    return "\n".join([
        report.comment(rec),
        "",
        f"Request and explanation PRs: {linked}. After this merges, update the side's `verification` in "
        f"`{explanation_path(rec['request'])}` and cite this PR. See EXPERIMENTS.md, \"Results\".",
    ])


def publish(records, run=sh, now=None):
    """One results PR per record. Returns ({request id: results PR url}, [failure messages])."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    start = run("git", "rev-parse", "--abbrev-ref", "HEAD").strip()
    base = run("git", "rev-parse", "HEAD").strip()
    existing = remote_branches(run)
    prs = all_prs(run)
    urls, failures = {}, []
    for rec in records:
        rid = rec["request"]
        branch = branch_name(rid, existing, now)
        try:
            run("git", "checkout", "-q", "-B", branch, base)
            (ROOT / "results" / f"{rid}.yaml").write_text(report.dump(rec))
            run("git", "add", f"results/{rid}.yaml")
            run("git", "commit", "-q", "-m", title(rec))
            run("git", "push", "-q", "-u", "origin", branch)
            targets = link_targets(rec, prs, run)
            url = run("gh", "pr", "create", "-R", REPO, "--base", "main", "--head", branch,
                      "--title", title(rec), "--body", pr_body(rec, targets)).strip()
            urls[rid] = url
            existing.add(branch)
            for n in targets:
                run("gh", "pr", "comment", str(n), "-R", REPO, "--body", report.comment(rec, url))
            print(f"{rid}: {url} (linked from {', '.join(f'#{n}' for n in targets) or 'no PR'})")
        except (subprocess.CalledProcessError, OSError) as e:
            detail = getattr(e, "stderr", "") or str(e)
            failures.append(f"{rid}: {detail.strip()[:300]}")
            print(f"{rid}: FAILED {failures[-1]}")
        finally:
            run("git", "checkout", "-q", "-f", start if start != "HEAD" else base)
    return urls, failures
