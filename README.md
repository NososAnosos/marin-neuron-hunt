# Marin Neuron Hunt

A [Station House](https://app.station-house.net) community project to explain as many of
[Marin 32B](https://huggingface.co/marin-community/marin-32b-base)'s MLP neurons as possible, and to back
each explanation with an experiment.

Marin 32B has 64 layers × 27,648 MLP neurons = 1,769,472 neurons. Each neuron's activation is signed, so every
neuron has a **positive side** and a **negative side**, and an explanation must cover both.

The data to investigate is the
[Marin 32B Neuron Atlas](https://huggingface.co/datasets/Station-house/marin-32b-neuron-atlas) on Hugging Face:
per-neuron statistics, top exemplars, logit lens, every strong firing, and (for a smaller sample) the residual
stream and dense activations.

## How it works

1. **Pick a neuron.** Search this repo and its open PRs first so you don't duplicate someone's work.
2. **Investigate it** with whatever tools you like: DuckDB straight from the Hub, pandas, notebooks, your own
   models or agents. Agent-written explanations are welcome; say which model or agent you used.
3. **Open a PR** that adds `explanations/L<layer>/<neuron>.yaml` (both sides; see [the format](#the-explanation-format))
   and, if you want it tested, a confirmation-experiment request in `requests/`. Choose one from
   [the experiment menu](EXPERIMENTS.md) or design your own.
4. **Review.** A maintainer checks the format and that the request is runnable, then merges.
5. **Daily run.** Merged requests are run in a daily batch. Each result goes in its own results PR that adds
   `results/<request id>.yaml`, linked from a comment on your PR. The side's `verification` then becomes
   `confirmed`, `refuted` or `inconclusive`.

> **Status (November 2026):** the daily experiment runner is live. It runs merged requests every day at
> 21:00 UTC and opens one results PR per request (see [EXPERIMENTS.md](EXPERIMENTS.md#results)).

## Which data to use

| prefix | use it for |
|---|---|
| `pilot/fineweb-10m-sparse/` | **Start here.** 10M tokens, 4,883 docs: exemplars, stats, every strong firing. |
| `pilot/fineweb-1m/` | Residual-stream and dense (`deep`) vectors, and token text (`tokens.parquet`). |
| `pilot/stage-a-100k/` | Smoke test only. |

The pilots are independent samples: `doc_id` and `token_idx` don't join across prefixes. Always say which
prefix your evidence comes from. The dataset card has the schema and example queries.

```sql
-- strongest firings of L40-N25546's positive side, with the firing token
SELECT e.rank, e.act, t.token_str
FROM 'hf://datasets/Station-house/marin-32b-neuron-atlas/pilot/fineweb-1m/summaries/exemplars.parquet' e
JOIN 'hf://datasets/Station-house/marin-32b-neuron-atlas/pilot/fineweb-1m/summaries/tokens.parquet' t USING (token_idx)
WHERE e.layer = 40 AND e.neuron = 25546 AND e.pole = 1 ORDER BY e.rank LIMIT 10;
```

## Naming

- Layers are numbered **0–63**, neurons **0–27647**.
- A neuron is `L40-N25546`. One side is `L40-N25546+` (positive) or `L40-N25546-` (negative), written with
  ASCII `+` and `-`.
- File paths use the neuron: `explanations/L40/L40-N25546.yaml`.

## The explanation format

One YAML file per neuron, checked against [`schema/explanation.schema.json`](schema/explanation.schema.json).
[`examples/L40-N25546.yaml`](examples/L40-N25546.yaml) shows every field. It's a format example, not an explanation:
L40-N25546 is still unexplained.

| field | meaning |
|---|---|
| `neuron` | e.g. `L40-N25546` |
| `data` | the atlas prefix your evidence comes from |
| `label` | at most 8 words for what the neuron does overall |
| `positive`, `negative` | one block per side (below); **both are required** |
| `relation` | how the two sides relate: `variant`, `cue-vs-absence`, `lumped`, `unrelated` or `none` |
| `relation_note` | one sentence on that relation |
| `by` | who wrote it: GitHub handle, `method: human` or `method: agent`, and the model or agent if any |

Each side has:

| field | meaning |
|---|---|
| `status` | `explained`, `unclear`, `silent` or `no-own-job` |
| `confidence` | `high` or `weak` (for `explained` only) |
| `cue_type` | `lexical`, `subword`, `grammar`, `phrase-idiom`, `semantic-topic`, `discourse-structure`, `format-markup`, `code`, `numeric-id`, `next-token`, `mixed` or `none` |
| `explanation` | one or two sentences |
| `fires_on` / `not_on` | "fires on X but not on Y" |
| `evidence` | exemplar ranks (from `exemplars.parquet`) or `token_idx` values that support it |
| `verification` | `untested`, `requested`, `confirmed`, `refuted` or `inconclusive`. Only an experiment result changes it. |

What the statuses mean:

- **`explained`**: the side has a clear job you can state.
- **`unclear`**: it fires, but you can't name a pattern. This is a valid answer, and it tells others the neuron
  still needs work.
- **`silent`**: the side barely fires **on this corpus**. That doesn't mean it's dead: the pilots are long
  English web documents, so code, math and other languages are barely covered.
- **`no-own-job`**: the side's firings are just the absence or mirror of the other side's cue (for example, small
  negative values wherever the positive cue is missing), with no pattern of their own.

What the relations mean:

- **`variant`**: both sides do versions of the same job.
- **`cue-vs-absence`**: one side marks a cue, the other its absence.
- **`lumped`**: one side does several unrelated jobs.
- **`unrelated`**: the sides do different jobs.
- **`none`**: one side is `silent` or `no-own-job`.

**Correlation isn't confirmation.** Top exemplars show where a side fires, not what it does. A claim that a
side does *not* fire on something can't be read from the `sparse` tier, which keeps only each token's 64
strongest neurons. Check it with the `deep` tier in `fineweb-1m`, or request an activation test.

## What counts

- A neuron is **explained** when a reviewed explanation file covering both sides is merged. `unclear` and
  `silent` sides count.
- A side is **confirmed** when a pre-registered experiment passes (see [EXPERIMENTS.md](EXPERIMENTS.md)).
  **Refuted** and **inconclusive** results are results too, and are credited.
- **First targets:** 1,000 neurons explained, and 100 sides confirmed.

## Disagreements and duplicates

- **One file per neuron.** If you disagree with a merged explanation, open a PR that adds your version under
  `alternatives:` in the same file, together with an experiment that would tell the two apart. The result
  decides which one becomes the main explanation.
- **Claiming a neuron:** claim it on Station House (the project's Neurons tab) so nobody else takes it while
  you work. One person holds a neuron at a time, and you can hold up to 5. A claim lapses after 7 days without
  activity; opening a PR for it stops the clock. A draft PR on an unclaimed neuron also claims it, once
  Station House sees it. Either way you need your GitHub account verified on your Station House profile.
- **Challenging an explanation:** an explained neuron can be claimed again by someone who thinks the
  explanation is wrong or incomplete. Their PR follows the alternatives rule above.
- **CI checks claims:** a PR that changes the explanation of a neuron someone else holds fails the `claims`
  check. List every neuron whose explanation your PR changes under `## Neuron(s)` (or in the title); that list
  is how Station House links the PR to your claim.

## Credit

Every explanation, request and result names its author in `by`. On Station House, a merged PR counts for a task
card when its title or description names the card's id: `marin-explain-neuron` for an explanation,
`marin-request-experiment` for a request. It's credited to the GitHub account verified on your Station House
profile. Credit will also show on the public coverage map once it launches.

## Repo layout

```
explanations/L<layer>/L<layer>-N<neuron>.yaml   one per neuron, both sides
requests/<neuron>-<nnn>.yaml                    confirmation-experiment requests
results/<neuron>-<nnn>.yaml                     written by the daily runner
requests/<request id>/DESIGN.md                 the design of a custom (type E) request
examples/                                       format examples (don't count as explanations)
schema/                                         JSON Schemas for the above
scripts/validate.py                             the check CI runs on every PR
scripts/check_design.py                         the design check CI runs on request PRs
```

Run the check locally: `pip install pyyaml jsonschema && python scripts/validate.py`.

How to design the evidence behind an explanation or request (controls, fresh prompts, noise) is in
[CONTRIBUTING.md](CONTRIBUTING.md). Notes for coding agents are in [AGENTS.md](AGENTS.md).

## License

Apache-2.0, the same as the atlas dataset. By opening a PR you agree to license your contribution under it.
