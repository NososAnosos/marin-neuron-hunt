## Neuron(s)

<!-- e.g. L40-N25546. List every neuron whose explanation this PR changes. Claim them on Station House first
(the project's Neurons tab); a PR on an unclaimed neuron claims it for you. CI fails if someone else holds one. -->

## In plain English

<!-- Required for any PR that adds or changes an explanation; CI fails without it. In a sentence or two per
neuron, say what you believe it does, for someone who hasn't read the YAML: no field names, no statistics.
Cover both sides, and say plainly when a side is unclear. With several neurons, start each with its id. -->

## What's in this PR

- [ ] An explanation: `explanations/L<layer>/L<layer>-N<neuron>.yaml`, covering **both sides**
- [ ] A confirmation-experiment request: `requests/<neuron>-<nnn>.yaml` (type: activation-test / steering / ablation / contrast / custom)
- [ ] An alternative to an existing explanation, plus the request that would tell them apart

## Station House credit

<!-- Keep the task id(s) this PR is for, and delete the rest. Credit goes to the GitHub account verified on your
Station House profile. -->

marin-explain-neuron · marin-request-experiment

## How you investigated

<!-- Data prefix, tools, queries or notebooks. If an agent wrote it, name the model or agent (also in `by`). -->

## Design

<!-- Required for a confirmation-experiment request (CI's `design` check fails without it); delete this
section for an explanation-only PR. One or two
lines each, written before any result. See CONTRIBUTING.md, "Designing a request". Type E: put it in
requests/<request id>/DESIGN.md instead and link it here. -->

### Claim

<!-- The side and the one-sentence prediction. -->

### Unit and metric

<!-- How many prompts per group; what's measured at the target token. -->

### Controls

<!-- Your near misses (same target token, wrong context), the random-neuron baseline, the other side. -->

### Data

<!-- Atlas prefix; prompts written fresh (not copied from exemplars); non-firing checked in `deep`. -->

### Noise

<!-- How far expected activations sit from the threshold; why this many prompts is enough. -->

### Readout

<!-- The pass rule, and the result that would refute the explanation. -->

## Checklist

- [ ] `python scripts/validate.py` passes
- [ ] I wrote the request's prompts and prediction before seeing any result for them
- [ ] The `## Design` section is filled in (the `design` check passes)
- [ ] Each marked `[[target]]` is a single token for the Marin tokenizer
