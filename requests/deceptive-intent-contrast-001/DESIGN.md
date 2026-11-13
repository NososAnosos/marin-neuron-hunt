# Deceptive-intent contrast screen: design and protocol

**Request:** `deceptive-intent-contrast-001` (type E, custom)
**Filed by:** the NososAnosos fleet (pairs authored by Hermia / Grok team, 2026-11-12; reviewed by Habibi,
2026-11-12; Kristopher Richards's go, 2026-11-12)
**Model:** marin-community/marin-32b-base @ f19b346615212ff9a6d92c10861a9cdb100cbf97

## Claim

Deceptive intent — a writer knowingly misleading a reader — is represented in Marin 32B by a small set of
MLP neurons: a single neuron or a coalition of at most 16 whose activation at the claim-completing word
separates enacted deception from honest persuasion, from narrated deception, and from sincere falsehood,
all at once. Registered prediction, made before any activation was seen on these prompts: no single
neuron passes all three contrasts; if anything passes, it is a coalition of 16 or fewer. A read-side pass
would claim representation only, not that the set decides anything; that stronger claim would need the
causal follow-on described under Readout.

## Unit and metric

The unit is one MLP neuron. The measurement is its activation, silu(gate) times up, at the single marked
target token of each prompt: 84 prompts, being 36 contrast pairs (12 per contrast, 72 texts) plus 12
topic-matched fillers, each text 11 to 28 Marin tokens. Per neuron and per contrast the metrics are
(a) the pair-direction rate, the share of the 12 pairs in which the high (deceptive) arm's activation
exceeds the low arm's, and (b) the mean high-minus-low activation difference across the 12 pairs.
Coalitions up to 16 units are scored greedily on summed activation with the same two metrics.

## Controls

- **Random-neuron baseline.** Per layer, 100 random neurons are scored with exactly the same
  pair-direction and mean-difference metrics as every candidate. The baseline's 95th percentile of mean
  difference is the bar a candidate must beat in each contrast, and if 5 or more of the 100 random
  neurons pass all three contrasts, the screen is too loose and the whole run is inconclusive.
- **Register control.** L20-N1234, the minced-oath / register neuron listed in the request, must fail all
  three contrasts. If it passes, the screen is reading register or urgency rather than intent, and the
  run is void.
- **Filler arm.** The 12 fillers share the pairs' topic vocabulary but carry no contrast. A candidate
  that also separates fillers from a nonexistent contrast is reading topic, not intent, and is rejected.

## Data

The evaluation set is the 84 hand-written prompts in `prompts.json`, written 2026-11-12 specifically for
this test against marin-community/marin-32b-base revision f19b346615212ff9a6d92c10861a9cdb100cbf97, and
tokenizer-checked against that revision: within every pair the marked target is the same Marin token
(single-token targets throughout). The set is fresh and held out by construction: the texts were written
before any activation was seen on them, are not copied from the atlas corpora or from any published
example, and no activation on them exists anywhere yet. Filters and cleaning: none. The texts are used
as written; the only exclusions ever applied were at authoring time (no real brands, links, phone
numbers, or operational fraud instructions). Label rules: pairs matched for topic, shared vocabulary,
and person/address where the contrast allows; a warning that quotes a lure is labeled by the writer's
act (honest), never as enacted deception; narrated arms carry varied frames (news, court complaint,
case study, investigation memo, documentary, oral history, police blotter, novel, historical records) so
no single fiction cue solves contrast 2; every sincere-falsehood arm is confident and uncorrected, so
there is no apology or hedging tell. An atlas-first route was tried and is documented here because it
shaped this design: mining the FineWeb pilots yielded 1 enacted scam text, 4 honest warnings, and 0
clear honest mistakes, so the corpus could not nominate candidates and two of the three contrasts could
not be run from it at all.

## Noise

The readout itself is deterministic: one forward pass per prompt on a fixed model revision, no sampling,
no generation, so rerunning the identical request returns the identical activations. The real noise is
item sampling: 12 pairs per contrast is a small set, and a neuron can look separating by luck. That noise
floor is estimated empirically, per layer, by the random-neuron baseline distribution (100 random neurons
scored identically): a candidate counts only where its mean difference clears the baseline's 95th
percentile, and the 80 percent pair-direction rule (at least 10 of 12 pairs in the predicted direction)
is the second, independent guard against a few large pairs carrying the mean. Coalition scores inherit
the same bars; no coalition is reported without the random-baseline comparison at its own layer.

## Readout

A candidate passes only if it clears all three contrasts: pair-direction rate at least 80 percent
(at least 10 of 12 pairs) in each contrast, and mean difference above the 95th percentile of the
random-neuron baseline in each contrast. The headline result is the smallest passing set, single neuron
or coalition up to 16. The claim is refuted, and will be reported as a clean null, if no single neuron
and no coalition up to 16 clears all three contrasts while the controls behave (L20-N1234 failing, fewer
than 5 of 100 random neurons passing). That null would mean deceptive intent is not a small localized
MLP coalition recoverable by this method, or that these labels do not isolate it; it would not prove
Marin has no representation of deception, since the feature could live in attention, in a residual
direction, or in a larger set. A read-side pass licenses only a follow-on causal request (type B for a
single neuron, type E for a coalition; log-probability at a marked slot against the same random baseline,
on short already-written frames, never generated lures) before any decider language, on the precedent of
L20-N1234-004. The fleet publishes its scoring either way, pass or null.

## Declared confounds (seen before filing, stated so they are on record)

- **Contrast 1 length asymmetry.** Honest arms average somewhat longer than lures (they explain the
  safe path). Readout is at a single matched token, and contrasts 2 and 3 do not share this pattern
  (contrast 3 pairs are near length-matched), so a pure length artifact would also have to survive
  contrast 3 to pass. We accept the residual risk rather than pad lures with filler text.
- **Contrast 1 negation density.** Honest notices naturally negate the lure's ask ("do not send",
  "will not ask"). Some honest arms are affirmative instead, and again the three-contrast gate is the
  protection: a negation feature has no purchase in contrast 3, where both arms assert the claim.
- **Contrast 3 legibility.** Intent must be legible in text to be tested at all; the knowledge state is
  carried in each arm's prefix clause by design. That is the operationalization, stated openly.

## Why this is a custom request and not a menu contrast test

The menu's contrast type (D) tests a named neuron. This filing has no named candidate: the atlas screen
that would have nominated one was run first and failed at the data layer (see Data). What the runner is
asked for is the measurement only: 84 forward passes of at most 28 tokens, no generation, returning the
full MLP activation table at each marked token. Scoring happens locally with the same machinery as the
fleet's capybara pair screen, under the bars fixed above.
