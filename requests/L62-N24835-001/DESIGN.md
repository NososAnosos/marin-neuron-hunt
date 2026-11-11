# L62-N24835-001: capybara scene control (animal or spelling?)

Request file: requests/L62-N24835-001.yaml. Type E (custom), because the
question is a comparison across ten animals and many readouts at once, scored with frozen group readouts.
No menu type covers a group readout or a ten-way rival margin. It is proposed for the maintainers to run
because the 32B model does not fit on our 8 GB GPU. Frozen readouts: readouts.json (20 named
readouts, 1,000 random sets). Prompts: prompts.json. The screen write-up is negative-result.md.

### Claim

The capybara candidates from probes/capybara-v1 (four ranked singles, L50-N4018-, and the best capped
groups of 4 to 16 sides) separate capybara from nine rival animals only when the word is written, not when
a capybara scene is described without it. If any readout does separate the unnamed scene, that readout
tracks the animal, not the tokens cap y bara.

### Unit and metric

Unit: one prompt with one marked single-token target. 182 prompts: scene arm 120 (10 animals x 3
descriptions x 4 shared endings, target " it"), word arm 40 (10 animals x 4 frames, target = last name
piece), ruby arm 12, unrelated 10. Metric: each readout's score at the target and the next token; margin
of the target animal over the strongest rival, in rival-SD units, as in the coalition screens.

### Controls

Baseline: 200 random sets per size (1, 4, 10, 12, 16) from the 1,920 random neurons (30 per layer, all
64 layers) extracted for the coalition screens, default_rng(0), readouts fit the same way and frozen in
readouts.json before the run; S3 needs the 99th percentile. Near misses: nine rival animals,
four of them South American rodents or near neighbours (guinea pig, nutria, mara, tapir), described with
overlapping facts (rodent, South America, rivers, orange teeth). Token-only control: the ruby arm, where
the gem context comes before the word. Fake target: the tapir groups, chosen on word spelling, scored for
tapir on the same scene arm; they are expected to fail. The other side: both poles of the five singles are
recorded and reported.

### Data

Selection data: probes/capybara-v1 on Station-house/marin-32b-neuron-atlas @ 6402b9e (20 pair templates,
cached dense activations). All 182 prompts are fresh. None is copied from probe templates or atlas
exemplars, and the scene descriptions avoid the English, Spanish and Portuguese names and their pieces.
Every target was checked as one Marin token with runner/prompts.py (tokenizer @ f19b3466); longest prompt
48 tokens. Artifacts: the scene target " it" is identical across animals, so only the description differs.

### Noise

12 scene prompts per animal; S2 allows two lost slots. Twelve capybara readouts are tested, so the random
baseline uses the 99th percentile rather than the 95th. Held-out margins on the cached data were thin at
the next token (0.004 to 0.08 rival-SD on the worst template), so a single prompt can flip a slot; S1 is
on the mean and S2 on slots for that reason.

### Readout

Per readout: confirmed if V1 (word margin >= 1.0), V2 (tapir sum-4 set fails S1 and S2 for tapir), S1
(scene margin >= 1.0), S2 (>= 10/12 slots), S3 (> 99th percentile of random sets), C1 (ruby fraction
<= 0.35 at bara and next) and C2 (worst ruby prompt <= 0.50) all hold. Refuted (spelling-bound) if V1 and
V2 hold but the scene margin is < 0.5 or slots < 6/12. Reported as a South American mammal cluster if the
top five are capybara, tapir, mara, nutria and guinea pig. Inconclusive if V2 fails or nothing passes V1.
If refuted, the PR's conclusion stands as is: no capybara neuron or group in these candidates.
