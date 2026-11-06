# Confirmation experiments

Every request names **one neuron side** (or a set of neurons), states a **prediction**, and fixes a **pass
rule** before anything runs. Write your prompts and prediction before you see any result: the daily runner uses
exactly what was merged.

Every request needs a design, and CI fails a request PR without one. Before writing one, work through the six design stages in [CONTRIBUTING.md](CONTRIBUTING.md#designing-a-request):
the claim, what's measured, the controls (near misses, random neurons), fresh prompts, noise and the
result that would refute you.

Pick a type from the menu, or design your own (type E). Copy the template into
`requests/<neuron>-<nnn>.yaml`, where `<nnn>` numbers the requests for that neuron (`001`, `002`, …).
[`examples/L40-N25546-001.yaml`](examples/L40-N25546-001.yaml) is a complete example.

All runs use Marin 32B (`marin-community/marin-32b-base` @ `f19b3466`, bf16), the atlas's definition of
activation (`act = silu(gate) * up`), and the atlas's per-neuron thresholds from
`pilot/fineweb-10m-sparse/summaries/pole_stats.parquet`.

In prompts, mark the **target token** with `[[ ]]`, e.g. `...her husband disagrees[[ or]] vice versa`. The
marked text must be exactly one token for the Marin tokenizer; the validator can't check that, but the runner
rejects the request if it isn't.

## Menu

### A. Activation test: "does it fire where I say, and not where I say it doesn't?"

The cheapest test, and the right first test for most explanations.

- Write **at least 10 prompts that should fire the side** and **at least 10 that shouldn't**.
- Good shouldn't-fire prompts are near misses: the same words in a context where your explanation says the side
  stays quiet.
- **Measured:** the side's activation at each target token.
- **Threshold:** the side's 99th percentile on the atlas (`pos_q99` for `+`, `neg_q99` for `-`), unless you
  set a number.
- **Pass:** at least 80% of the should-fire prompts exceed the threshold **and** at least 80% of the
  shouldn't-fire prompts stay below it. The runner also reports AUROC.

```yaml
type: activation-test
neuron: L40-N25546
side: "+"
hypothesis: "..."
should_fire: ["... [[target]] ...", ...]     # >= 10
should_not_fire: ["... [[target]] ...", ...] # >= 10
threshold: atlas-q99        # or a number
by: {github: your-handle, method: human}
```

### B. Steering: "if I push it, does the model do what I predict?"

- The side is set to **k × its scale** (`scale_abs_p999`) at the target token. Default k = 4; `+` pushes it
  positive, `-` negative.
- Name the **tokens you expect the next-token prediction to move toward** (and, optionally, away from), and
  write **at least 10 prompts**.
- **Measured:** the change in log-probability of your named tokens, compared with the same steering applied to
  **100 random neurons in the same layer**.
- **Pass:** your tokens' mean log-probability gain beats the 95th percentile of the random-neuron baseline, in
  the predicted direction.

```yaml
type: steering
neuron: L40-N25546
side: "+"
k: 4
prompts: ["... [[target]]", ...]   # >= 10; steering applies at the marked token
expect_up: [" vice", " Conversely"]
expect_down: []
by: {github: your-handle, method: agent, model: "..."}
```

### C. Ablation: "if I remove it, does something specific break?"

- The neuron is set to zero (or to its mean, `mode: mean`) at the target token on **at least 10 prompts where
  it fires**. A set of neurons can be ablated together.
- Name a **metric** (the log-probability of a named next token, or the loss on your prompts) and a **predicted
  effect size**.
- **Measured:** the metric change, compared with ablating **100 random neurons in the same layer** (sets of the
  same size, for a set).
- **Pass:** the effect beats the 95th percentile of the random baseline in the predicted direction and reaches
  at least half your predicted size.
- **Read this before choosing ablation:** knocking out one neuron in a 32B model often changes almost nothing,
  because other neurons do the same job. A null result is reported as **inconclusive**, not refuted. Ablation is
  most informative on a set of neurons you think are redundant.

```yaml
type: ablation
neurons: [L40-N25546]   # one or more
mode: zero              # or mean
prompts: ["... [[target]] ...", ...]  # >= 10
metric: {kind: next-token-logprob, token: " vice"}   # or {kind: loss}
predicted_effect: -1.0  # change in the metric, in nats
by: {github: your-handle, method: human}
```

### D. Contrast test: "does the one difference I name drive it?"

- Write **at least 10 pairs** of prompts that differ only in the thing your explanation says matters (for
  example "or vice versa" versus "or the reverse"), with the target token marked in both.
- **Measured:** the side's activation difference within each pair.
- **Pass:** the side is higher on the predicted member in at least 80% of pairs, and the mean difference is at
  least half the side's threshold.

```yaml
type: contrast
neuron: L40-N25546
side: "+"
pairs:
  - {high: "... [[target]] ...", low: "... [[target]] ..."}   # >= 10 pairs
by: {github: your-handle, method: human}
```

### E. Your own experiment

Anything not on the menu: activation patching, interactions between neurons, a new prompt set, a different
metric, attention attribution, and so on.

- Write the design in `requests/<request id>/DESIGN.md` (`stationhouse design new requests/<request id>/DESIGN.md`).
- Describe **the procedure** precisely enough that someone else could run it.
- State **the prediction**, **the pass rule**, and what result would **refute** you.
- Estimate **the compute** (prompts × forward passes) and say what the runner should save.
- Code is welcome in the PR (under `requests/<request id>/`), but a maintainer reviews it before it runs. Custom
  experiments may take longer than one daily batch.

```yaml
type: custom
neurons: [L40-N25546]
title: "..."
procedure: |
  ...
prediction: "..."
pass_rule: "..."
refuted_if: "..."
compute: "about N forward passes of up to M tokens"
outputs: "..."
by: {github: your-handle, method: human}
```

## Results

The runner writes `results/<request id>.yaml` with the measured numbers, `outcome: confirmed | refuted |
inconclusive`, the model commit and the date.

**One results PR per request.** Each result gets its own PR, titled `Results <request id>: <outcome>`, that
commits only `results/<request id>.yaml`. The runner then comments on the PR that added the request (and on
the explanation PR, if that's a different one), linking the results PR. A rerun of the same request replaces
its file in a new results PR.

**Requests merge before they run.** A request runs only once it is on `main`. If the explanation in the same
PR isn't ready yet, a maintainer can merge the request files in a PR of their own, keeping the author's
credit, so the experiment doesn't wait on the explanation review.

After a results PR merges, the explanation author or a maintainer updates the side's `verification` field
in the explanation and cites the results PR. After a `refuted` result, also revise the explanation to fit
the numbers, or set the side to `unclear`.

## Limits

- At most **3 open requests per contributor** at a time. Each daily batch runs requests in the order they were
  merged, up to the day's compute cap.
- Prompts are at most 512 tokens. Longer ones go in as type E.
- Requests built to make the model produce harmful content are closed.
