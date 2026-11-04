# Contributing to the Marin Neuron Hunt

The [README](README.md) covers the loop, from claiming a neuron to merging an explanation and
requesting an experiment. It also covers the file formats. This page covers **designing the
evidence**: the investigation behind an explanation, and the confirmation experiment that tests it.
Most weak explanations come from a design problem. The pattern holds for most of the strongest
firings of a neuron, so it reads as real. Nobody checked whether random neurons in the same layer
show the same pattern, or whether the same token fires in a context the explanation says it
shouldn't.

## Designing a request

Every request has the same six stages, whatever its type in [EXPERIMENTS.md](EXPERIMENTS.md). Settle
them before you write any prompts.

1. **Claim.** One side, one sentence: "`L40-N25546+` fires on X but not on Y", or "pushing it moves
   the next token toward Z". This becomes the request's `hypothesis` or `prediction`.
2. **Unit and metric.** The unit is one prompt with one marked target token. The metric is the
   side's activation at that token for A and D, or the log-probability change of named tokens for
   B and C. Say how many prompts. The minimum is 10 per group, but 20 to 30 tells you much more.
3. **Controls.**
   - **Near misses:** the same target token in a context where your explanation says the side
     stays quiet. These are what separate "fires on the word *or*" from "fires on *or* in *or vice
     versa*". If every one of your should-fire prompts shares a token and none of your
     shouldn't-fire prompts do, you're testing the token, not your explanation.
   - **Random neurons:** steering and ablation are compared with 100 random neurons in the same
     layer. For an explanation, look at a few random neurons from the layer as well. If their top
     exemplars show your pattern too (document starts, mid-word name pieces, punctuation), the
     pattern isn't specific to your neuron.
   - **The other side:** say what the opposite side does on the same prompts.
4. **Data.**
   - Name the atlas prefix your evidence comes from.
   - Write fresh prompts. Don't copy exemplar text from the atlas into the request: those are the
     examples you built the hypothesis from, so they will pass whether or not it's right.
   - The `sparse` tier can't show where a side *doesn't* fire. Check non-firing claims in the
     `deep` tier of `fineweb-1m`.
   - Watch for artifacts: the first tokens of a document, document boundaries, repeated
     boilerplate, and subword pieces in the middle of a word.
   - Every `[[target]]` must be a single Marin token.
5. **Noise.**
   - With 10 prompts, the 80% rule allows two misses per group, so a borderline result means
     little. Write more prompts if your explanation is weak.
   - Look at how close your expected activations are to the threshold (`pos_q99` / `neg_q99`). A
     side that sits right at the threshold will flip on small wording changes.
   - A null ablation is reported as `inconclusive`, because other neurons often do the same job.
     Plan for that, or ablate a set of neurons.
6. **Readout.** Use the menu's pass rule, or write your own for type E. Say before the run which
   result would refute your explanation, and what you would change in the explanation if it did.
   A `refuted` result is credited and is useful to everyone.

**Where the design goes, and the check.** A design is required for every request. For types A to D,
fill in the `## Design` section of the PR template, keeping it to one or two lines per stage. For
type E, write `requests/<request id>/DESIGN.md` from the [template](#tools), next to your code.

The `design` CI check fails any PR that adds or changes a request without one. Each of the six
stages must be filled in, and Controls must name a baseline. Editing the PR description re-runs the
check. Run it locally with `python scripts/check_design.py <file>`. A maintainer still reads the
design before merging: passing the check means nothing is missing, not that the design is good.
After the run, leave the design as it was, and put anything that changed in the PR.

**Order of tests.** Start with an activation test (A) or a contrast test (D). They are cheap, and a
failure tells you exactly which prompts broke. Move on to steering (B) or ablation (C) only once the
side fires where you say it does.

### Before you open the PR: the Experimentation Wizard

Once the design is written, run it through the Station House Experimentation Wizard: three rounds
of reviewer questions on your design, which you answer yourself.

1. **The claim and the controls**
2. **The data and the noise**
3. **The metric and the readout**

Revise the design between rounds. When you finish, you get a `Design-Session: ds_…` line: put it in
the PR description.

To run it:
- **Claude Code:** install the plugin (`/plugin marketplace add Station-House/stationhouse-cli`,
  then `/plugin install stationhouse@stationhouse-cli`) and ask it to open the PR. The questions
  appear as a form in Claude Code, and the plugin won't push a request whose design hasn't been
  reviewed.
- **Other agents:** `stationhouse mcp`, plus `stationhouse hooks install` for the same push check.
- **Without MCP:** `stationhouse design start "<title>"`, then `stationhouse design round <file>`
  for each round.

A request PR without the line gets a warning in CI and a note on the Station House platform. Both
are notes, not blocks.

## Tools

```bash
pip install git+https://github.com/likelytobelaura/stationhouse-cli
stationhouse design new requests/L40-N25546-002/DESIGN.md   # the six-stage template (type E)
stationhouse design check requests/L40-N25546-002/DESIGN.md # missing pieces (no model)
stationhouse design review requests/L40-N25546-002/DESIGN.md  # a critique from the Station House assistant
```

`stationhouse design check` uses the same rules as the CI check. It also works on a PR description saved to a file, as long as it has the six `### ...`
headings under `## Design`. It finds missing pieces, not bad ideas. For the thinking, ask in the
[atlas discussions](https://huggingface.co/datasets/Station-house/marin-32b-neuron-atlas/discussions),
use `design review`, or use a coding agent. Claude Code, Codex and Cursor read [AGENTS.md](AGENTS.md)
in this repo.

## Agents

Agent-written explanations and requests are welcome. Say which model or agent you used in `by`.
You're still the author, so check the evidence yourself. [AGENTS.md](AGENTS.md) tells agents how to
help you design a request without designing it for you.
