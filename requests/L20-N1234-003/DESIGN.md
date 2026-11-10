# L20-N1234-003: strong vs soft exclamations in matched frames

DRAFT (local only). Request file: requests/L20-N1234-003.yaml. Type E (custom), because the question is
whether two arms are about equal, and no menu type has an equivalence pass rule: contrast (D) passes only
when one member is higher, and an activation test (A) checks only threshold crossings.

### Claim

`L20-N1234+` treats strong profanity and its minced forms as parallel shapes ("god damn it" ~ "gosh darn
it"): it fires above pos_q99 on both, at comparable strength, and not on neutral words in the same slot.
This is the primary falsification test of the minced-oath-specific reading in the explanation.

### Unit and metric

Unit: one prompt with one marked single-token target. 12 frames x 3 arms (strong S, minced M, neutral N) =
36 prompts. Metric: positive-side activation silu(gate)*up at the target; per-frame ratio r = S/M with both
floored at T/4.

### Controls

Neutral arm N: the same frame with a non-oath word in the slot (the near misses). Random-neuron baseline:
100 random L20 neurons (seed 0) scored on the same prompts with their own pos_q99; fewer than 5 may pass.
The other side: negative-side activation is recorded and expected near zero on all arms.

### Data

Thresholds from pilot/fineweb-10m-sparse pole_stats (pos_q99 = 0.2480); model marin-32b-base @ f19b3466.
All 36 prompts are fresh, written before any activation was measured on them and not copied from atlas
exemplars. Filters: none. Every target checked as a single Marin token (tokenizer.json @ f19b3466); gosh,
goddamn, Jeez, Geez, fudge, frick, frickin and frigging split, so they were replaced.

### Noise

Expected firings (atlas exemplars for this cluster sit at about 1.6 to 2.9) are well clear of the 0.2480
threshold, but the threshold is low, so the neutral arm may also cross it. 12 frames give the 80% rule two
misses per arm; the random-neuron baseline estimates how often the rules pass by chance.

### Readout

Confirmed if M > T in >= 10/12, S > T in >= 10/12, N > T in <= 2/12, geometric-mean r in [0.5, 2] with
>= 8/12 frames in that band, and < 5/100 random neurons pass. Refuted (minced-specific) if S > T in < 6/12
or geometric-mean r < 0.5; refuted the other way (general profanity) if r > 2. Inconclusive if the minced
or neutral rule fails, or the random baseline passes too often.
