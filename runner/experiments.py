"""The pass rules in EXPERIMENTS.md, one function per request type. Each returns a dict with `outcome`
(confirmed / refuted / inconclusive), `reason`, `summary` (the numbers the pass rule uses) and `per_prompt`.
"""

import hashlib
import re

import numpy as np

from runner.model import Row
from runner.prompts import encode, single_token

PASS_FRACTION = 0.8
BASELINE_N = 100
DEFAULT_K = 4


def parse_neuron(name):
    m = re.fullmatch(r"L(\d+)-N(\d+)", name)
    return int(m.group(1)), int(m.group(2))


def side_sign(side):
    return 1.0 if side == "+" else -1.0


def side_threshold(stats, side):
    """neg_q99 is stored as a magnitude, so thresholds are compared against sign * activation."""
    return stats["pos_q99"] if side == "+" else stats["neg_q99"]


def seed_for(request_id):
    return int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8], "little")


def auroc(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def random_neurons(rng, n_neurons, k, exclude):
    pool = np.setdiff1d(np.arange(n_neurons), np.asarray(list(exclude), dtype=np.int64))
    return rng.choice(pool, size=k, replace=False).tolist()


def result(outcome, summary, per_prompt, reason=None, **extra):
    return dict(outcome=outcome, reason=reason, summary=summary, per_prompt=per_prompt, **extra)


def activation_test(m, req, request_id):
    layer, neuron = parse_neuron(req["neuron"])
    stats = m.stats(layer, neuron)
    sign = side_sign(req["side"])
    thr = req.get("threshold", "atlas-q99")
    thr = side_threshold(stats, req["side"]) if thr == "atlas-q99" else float(thr)
    fire = [encode(m.tok, t) for t in req["should_fire"]]
    quiet = [encode(m.tok, t) for t in req["should_not_fire"]]
    acts, _, _ = m.run([Row(e.ids, e.pos) for e in fire + quiet], reads=[(layer, neuron)])
    a = acts[:, 0]
    fa, qa = a[:len(fire)], a[len(fire):]
    fire_above = float(np.mean(sign * fa > thr))
    quiet_below = float(np.mean(sign * qa <= thr))
    per = [dict(group=g, prompt=e.text, token=e.token, act=float(v))
           for g, es, vs in (("should_fire", fire, fa), ("should_not_fire", quiet, qa)) for e, v in zip(es, vs)]
    ok = fire_above >= PASS_FRACTION and quiet_below >= PASS_FRACTION
    summary = dict(threshold=thr, should_fire_above=fire_above, should_not_fire_below=quiet_below,
                   auroc=auroc(sign * fa, sign * qa), should_fire_mean=float(fa.mean()),
                   should_not_fire_mean=float(qa.mean()))
    return result("confirmed" if ok else "refuted", summary, per, atlas_stats=stats)


def contrast(m, req, request_id):
    layer, neuron = parse_neuron(req["neuron"])
    stats = m.stats(layer, neuron)
    sign = side_sign(req["side"])
    thr = side_threshold(stats, req["side"])
    enc = [(encode(m.tok, p["high"]), encode(m.tok, p["low"])) for p in req["pairs"]]
    acts, _, _ = m.run([Row(e.ids, e.pos) for pair in enc for e in pair], reads=[(layer, neuron)])
    hi, lo = acts[0::2, 0], acts[1::2, 0]
    diff = sign * (hi - lo)
    frac = float(np.mean(diff > 0))
    mean_diff = float(diff.mean())
    per = [dict(high=h.text, low=l.text, act_high=float(a), act_low=float(b), diff=float(d))
           for (h, l), a, b, d in zip(enc, hi, lo, diff)]
    ok = frac >= PASS_FRACTION and mean_diff >= 0.5 * thr
    summary = dict(threshold=thr, pairs_in_predicted_direction=frac, mean_diff=mean_diff,
                   mean_diff_needed=0.5 * thr)
    return result("confirmed" if ok else "refuted", summary, per, atlas_stats=stats)


def steering(m, req, request_id):
    layer, neuron = parse_neuron(req["neuron"])
    sign = side_sign(req["side"])
    k = float(req.get("k", DEFAULT_K))
    up = [single_token(m.tok, t) for t in req["expect_up"]]
    down = [single_token(m.tok, t) for t in req.get("expect_down") or []]
    enc = [encode(m.tok, t) for t in req["prompts"]]
    rng = np.random.default_rng(seed_for(request_id))
    baseline = random_neurons(rng, m.n_neurons, BASELINE_N, {neuron})
    tokens = up + down

    def effect(lp, clean):
        d = lp - clean  # [prompts, tokens]
        e = d[:, :len(up)].mean(axis=1)
        if down:
            e = e - d[:, len(up):].mean(axis=1)
        return e  # per prompt

    clean_rows = [Row(e.ids, e.pos) for e in enc]
    steer = lambda n: [Row(e.ids, e.pos, [(layer, n, sign * k * m.stats(layer, n)["scale_abs_p999"])]) for e in enc]
    rows = clean_rows + steer(neuron) + [r for n in baseline for r in steer(n)]
    acts, lps, _ = m.run(rows, reads=[(layer, neuron)], tokens=tokens)
    p = len(enc)
    clean = lps[:p]
    target_e = effect(lps[p:2 * p], clean)
    base = np.array([effect(lps[(2 + i) * p:(3 + i) * p], clean).mean() for i in range(len(baseline))])
    p95 = float(np.quantile(base, 0.95))
    score = float(target_e.mean())
    per = [dict(prompt=e.text, token=e.token, act=float(a), effect=float(x))
           for e, a, x in zip(enc, acts[:p, 0], target_e)]
    ok = score > 0 and score > p95
    summary = dict(k=k, set_to=sign * k * m.stats(layer, neuron)["scale_abs_p999"], effect=score,
                   baseline_p95=p95, baseline_mean=float(base.mean()), baseline_n=len(baseline),
                   effect_percentile=float(np.mean(base < score)))
    return result("confirmed" if ok else "refuted", summary, per, baseline_neurons=baseline)


def ablation(m, req, request_id):
    targets = [parse_neuron(n) for n in req["neurons"]]
    mode = req.get("mode", "zero")
    metric = req["metric"]
    predicted = float(req["predicted_effect"])
    direction = 1.0 if predicted >= 0 else -1.0
    enc = [encode(m.tok, t) for t in req["prompts"]]
    tok_id = [single_token(m.tok, metric["token"])] if metric["kind"] == "next-token-logprob" else None
    want_loss = metric["kind"] == "loss"
    rng = np.random.default_rng(seed_for(request_id))

    def value(layer, n):
        return m.stats(layer, n)["mean"] if mode == "mean" else 0.0

    def ablated(neurons):
        return [Row(e.ids, e.pos, [(l, n, value(l, n)) for l, n in neurons]) for e in enc]

    sets = []
    for _ in range(BASELINE_N):
        s = []
        for layer in sorted({l for l, _ in targets}):
            k = sum(1 for l, _ in targets if l == layer)
            taken = {n for l, n in targets if l == layer}
            s += [(layer, n) for n in random_neurons(rng, m.n_neurons, k, taken)]
        sets.append(s)
    rows = [Row(e.ids, e.pos) for e in enc] + ablated(targets) + [r for s in sets for r in ablated(s)]
    acts, lps, losses = m.run(rows, reads=targets, tokens=tok_id, loss=want_loss)
    metric_v = lps[:, 0] if tok_id else losses
    p = len(enc)
    clean = metric_v[:p]
    target_d = metric_v[p:2 * p] - clean
    base = np.array([np.nanmean(metric_v[(2 + i) * p:(3 + i) * p] - clean) for i in range(len(sets))])
    effect = float(np.nanmean(target_d))
    p95 = float(np.quantile(direction * base, 0.95))
    beats = direction * effect > p95
    big_enough = abs(effect) >= 0.5 * abs(predicted)
    opposite = -direction * effect > float(np.quantile(-direction * base, 0.95))
    if beats and big_enough:
        outcome = "confirmed"
    elif opposite:
        outcome = "refuted"
    else:
        outcome = "inconclusive"
    per = [dict(prompt=e.text, token=e.token, clean_acts=[float(x) for x in acts[i]], clean_metric=float(clean[i]),
                change=float(target_d[i])) for i, e in enumerate(enc)]
    summary = dict(mode=mode, metric=metric["kind"], predicted_effect=predicted, effect=effect,
                   baseline_p95_in_predicted_direction=p95, baseline_mean=float(np.nanmean(base)),
                   baseline_n=len(sets), beats_baseline=bool(beats), at_least_half_predicted=bool(big_enough))
    return result(outcome, summary, per, baseline_sets=[[f"L{l}-N{n}" for l, n in s] for s in sets[:5]])


RUNNERS = {"activation-test": activation_test, "contrast": contrast, "steering": steering, "ablation": ablation}
