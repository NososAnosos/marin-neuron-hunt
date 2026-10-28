"""Daily runner entry point.

    python -m runner.main --dry-run                 # what would run today, cost estimate, prompt check (CPU)
    python -m runner.main                           # run today's batch on Modal, write results/ locally
    python -m runner.main --publish                 # ... then open the results PR and comment on request PRs
    python -m runner.main --only L0-N10657-001      # just one pending request
    python -m runner.main --request-file x.yaml [y.yaml ...] --out dir/   # unmerged requests, results to dir/
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from runner import pick, report
from runner.prompts import check, load_tokenizer

REPO = os.environ.get("GITHUB_REPOSITORY", "Station-House/marin-neuron-hunt")


def sh(*args, **kw):
    return subprocess.run(args, check=True, capture_output=True, text=True, cwd=pick.ROOT, **kw).stdout


def in_open_results_prs():
    """Requests already answered in a results PR that hasn't merged yet."""
    prs = json.loads(sh("gh", "pr", "list", "-R", REPO, "--state", "open", "--limit", "100",
                        "--json", "headRefName,files"))
    return {Path(f["path"]).stem for pr in prs if pr["headRefName"].startswith("results-")
            for f in pr["files"] if f["path"].startswith("results/")}


def source_pr(commit):
    if not commit:
        return None
    try:
        prs = json.loads(sh("gh", "api", f"repos/{REPO}/commits/{commit}/pulls"))
    except subprocess.CalledProcessError:
        return None
    merged = [p for p in prs if p.get("merged_at")]
    return (merged or prs or [{}])[0].get("number")


def publish(records, date):
    branch = f"results-{date}"
    if sh("git", "ls-remote", "--heads", "origin", branch).strip():
        branch += datetime.datetime.now(datetime.timezone.utc).strftime("-%H%M")
    sh("git", "checkout", "-b", branch)
    sh("git", "add", *[f"results/{r['request']}.yaml" for r in records])
    outcomes = ", ".join(f"{r['request']} {r['outcome']}" for r in records)
    sh("git", "commit", "-m", f"Results {date}: {outcomes}")
    sh("git", "push", "origin", branch)
    body = ["Daily runner results. Merge after a look; then update each side's `verification` field.", ""]
    body += [f"- `{r['request']}` ({r['type']}): **{r['outcome']}**" for r in records]
    url = sh("gh", "pr", "create", "-R", REPO, "--base", "main", "--head", branch, "--title", f"Results {date}",
             "--body", "\n".join(body)).strip()
    print(url)
    for r in records:
        num = source_pr(r["request_commit"])
        if num:
            sh("gh", "pr", "comment", str(num), "-R", REPO, "--body", report.comment(r) + f"\n\nResults PR: {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--cap-usd", type=float, default=float(os.environ.get("RUNNER_CAP_USD", 10)))
    ap.add_argument("--request-file", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    if a.request_file:
        items = [pick.Pending(f.stem, f, yaml.safe_load(f.read_text()), 0, "") for f in a.request_file]
        custom, later = [], []
        a.publish = False
    else:
        exclude = in_open_results_prs() if a.publish or os.environ.get("GITHUB_ACTIONS") else set()
        queue = [p for p in pick.pending(exclude) if not a.only or p.id == a.only]
        items, custom, later = pick.batch(queue, a.cap_usd)

    print(f"run: {len(items)}, custom (maintainer): {len(custom)}, over cap: {len(later)}, "
          f"est. ${pick.est_cost(items):.2f}")
    for label, group in (("run", items), ("custom", custom), ("later", later)):
        for p in group:
            print(f"  {label:6s} {p.id} {p.request['type']} ~{pick.est_seconds(p.request):.0f}s")
    if a.dry_run:
        tok = load_tokenizer()
        bad = 0
        for p in items:
            for problem in check(tok, p.request):
                bad += 1
                print(f"  {p.id}: {problem}")
        print("prompt check:", "OK" if not bad else f"{bad} problem(s); those requests will come back inconclusive")
        return 0
    if not items:
        return 0

    import modal

    from runner.modal_app import app, run_batch

    payload = [dict(id=p.id, request=p.request, commit=p.commit) for p in items]
    with modal.enable_output(), app.run():
        results = run_batch.remote(payload)

    runner_commit = sh("git", "rev-parse", "HEAD").strip()
    date = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    out = a.out or pick.ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for item, res in zip(payload, results):
        rec = report.record(item, res, runner_commit, date, res.get("gpu", "?"))
        (out / f"{item['id']}.yaml").write_text(report.dump(rec))
        if a.request_file:
            (out / f"{item['id']}.comment.md").write_text(report.comment(rec))
        print(f"{item['id']}: {rec['outcome']}" + (f" ({rec['reason'][:200]})" if rec["reason"] else ""))
        records.append(rec)
    if a.publish:
        publish(records, date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
