"""Writes results/<request id>.yaml and the summary comment for the request's PR."""

import yaml

from runner.prompts import MODEL_REPO, MODEL_SHA

LABELS = {
    "threshold": "threshold",
    "should_fire_above": "should_fire above threshold",
    "should_not_fire_below": "should_not_fire below threshold",
    "auroc": "AUROC",
    "pairs_in_predicted_direction": "pairs in predicted direction",
    "mean_diff": "mean difference",
    "mean_diff_needed": "mean difference needed",
    "effect": "effect",
    "baseline_p95": "random-neuron baseline p95",
    "baseline_p95_in_predicted_direction": "random-set baseline p95 (predicted direction)",
    "effect_percentile": "effect percentile vs baseline",
}


def record(item, res, runner_commit, date, gpu):
    return dict(
        request=item["id"],
        type=item["request"]["type"],
        outcome=res["outcome"],
        reason=res.get("reason"),
        summary=res.get("summary") or {},
        per_prompt=res.get("per_prompt") or [],
        model=f"{MODEL_REPO}@{MODEL_SHA}",
        runner_commit=runner_commit,
        request_commit=item["commit"],
        gpu=gpu,
        date=date,
        seconds=round(res.get("seconds", 0.0), 1),
        **{k: res[k] for k in ("atlas_stats", "baseline_neurons", "baseline_sets") if k in res},
    )


def dump(rec):
    return yaml.safe_dump(rec, sort_keys=False, allow_unicode=True, width=120)


def fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def comment(rec, results_pr=None):
    """The summary: the body of the results PR (results_pr=None), or the comment on the request's PR."""
    lines = [f"## Experiment result: `{rec['request']}` → **{rec['outcome']}**", ""]
    if results_pr:
        lines += [f"Results PR: {results_pr}", ""]
    if rec.get("reason"):
        lines += [f"Reason: {rec['reason']}", ""]
    if rec["summary"]:
        lines += ["| | |", "|---|---|"]
        lines += [f"| {LABELS.get(k, k.replace('_', ' '))} | {fmt(v)} |" for k, v in rec["summary"].items()]
        lines.append("")
    if rec["per_prompt"]:
        keys = [k for k in rec["per_prompt"][0] if not isinstance(rec["per_prompt"][0][k], list)]
        lines += ["<details><summary>Per prompt</summary>", "", "| " + " | ".join(keys) + " |",
                  "|" + "---|" * len(keys)]
        for row in rec["per_prompt"]:
            cells = [fmt(row[k]).replace("|", "\\|").replace("\n", " ") for k in keys]
            lines.append("| " + " | ".join(f"`{c}`" if k in ("prompt", "high", "low", "token") else c
                                           for k, c in zip(keys, cells)) + " |")
        lines += ["", "</details>", ""]
    where = f"in {results_pr}" if results_pr else "in this PR"
    lines.append(f"Model `{rec['model']}` (bf16) on {rec['gpu']}, runner `{rec['runner_commit'][:7]}`, request "
                 f"`{rec['request_commit'][:7]}`, {rec['date']}. Full numbers: `results/{rec['request']}.yaml` "
                 f"{where}.")
    return "\n".join(lines)
