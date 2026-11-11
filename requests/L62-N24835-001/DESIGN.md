# L62-N24835-001: capybara scene control (animal or spelling?)

Request file: requests/L62-N24835-001.yaml. Type E (custom), because the
question is a comparison across ten animals and many readouts at once, scored with frozen group readouts.
No menu type covers a group readout or a ten-way rival margin. It is proposed for the maintainers to run
because the 32B model does not fit on our 8 GB GPU. Frozen candidate readouts: readouts.json (20 named
readouts, 1,000 random sets). Prompts: prompts.json, including the naming contexts for the positive
control. The screen write-up is negative-result.md.

### Claim

The capybara candidates from probes/capybara-v1 (four ranked singles, L50-N4018-, and the best capped
groups of 4 to 16 sides) separate capybara from nine rival animals only when the word is written, not when
a capybara scene is described without it. If any readout does separate the unnamed scene, that readout
tracks the animal, not the tokens cap y bara. A low candidate scene score counts as spelling-bound only
if the positive controls first show both that the descriptions evoke capybara in the model and that a
scene-fitted readout can separate held-out capybara descriptions at the same target.

### Unit and metric

Activation unit: one prompt with one marked single-token target. 182 prompts: scene arm 120 (10 animals
x 3 descriptions x 4 shared endings, target " it"), word arm 40 (10 animals x 4 frames, target = last
name piece), ruby arm 12, unrelated 10. Metric: each candidate readout's score at the target and the
next token; margin of the target animal over the strongest rival, in rival-SD units, as in the coalition
screens. Naming unit: 33 contexts (the 30 exact scene-description stems plus 3 neutral contexts) x 10
candidate names. Its score is the summed log probability of a candidate's full name after the context
and the frame " The animal is:". For candidate a and context d, lift is log P(a | d) minus the mean
log P(a | neutral context); using the same name in both terms subtracts its baseline under the naming
frame, including its ordinary frequency and token length. Scene-fit unit: the same scene activations at
" it", cross-fitted by description as specified under Controls; its held-out margin and matched slots
use the same definitions as the candidate readouts.

### Controls

Baseline: 200 random sets per size (1, 4, 10, 12, 16) from the 1,920 random neurons (30 per layer, all
64 layers) extracted for the coalition screens, default_rng(0), readouts fit the same way and frozen in
readouts.json before the run; S3 needs the 99th percentile. Near misses: nine rival animals,
four of them South American rodents or near neighbours (guinea pig, nutria, mara, tapir), described with
overlapping facts (rodent, South America, rivers, orange teeth). Token-only control: the ruby arm. All
12 ruby prompts and all 4 capybara word-arm prompts tokenize the name as " cap" + "y" + "bara", with
"bara" marked as the target; the only intended difference is the referent established by the context,
the Ruby testing library rather than the animal. Fake target: the tapir groups, chosen on word spelling, scored for
tapir on the same scene arm; they are expected to fail. The other side: both poles of the five singles are
recorded and reported.

Positive control P1, scene-fitted readout: fit a binary L2 logistic readout (C = 1.0, balanced class
weights) for capybara versus the nine rivals on the activations of all 1,960 recorded sides at " it".
Features are standardized inside each training fold. Cross-fit in 3 folds by description index: hold out
one description for every animal (40 prompts), train on the other two descriptions (80 prompts), and
pool the held-out scores. P1 passes if the pooled held-out capybara margin is >= 1.0 and capybara wins
>= 10 of 12 matched slots. The same cross-fitting is reported one-versus-rest for every animal as a
diagnostic of whether the scene arm carries animal identity in general. This readout is a control, not
a candidate, and it is never fitted on a description it is scored on.

Positive control P2, prior-corrected naming: score every candidate name after every naming context and
report both raw log probabilities and lifts. Raw probabilities are not a pass metric, because a common,
short name starts with a higher baseline than a rare, multi-token name. P2 passes for capybara if its
mean lift on its own 3 contexts is > 0, its lift ranks first among the 10 candidates in at least 2 of
those 3 contexts, and its mean lift on its own contexts is higher than its mean lift on the 27 rival
contexts. The true candidate's lift rank is also reported for all 30 animal contexts.

### Data

Selection data: probes/capybara-v1 on Station-house/marin-32b-neuron-atlas @ 6402b9e (20 pair templates,
cached dense activations). All 182 activation prompts are fresh. None is copied from probe templates or
atlas exemplars, and the scene descriptions avoid the English, Spanish and Portuguese names and their
pieces. Every target was checked as one Marin token with runner/prompts.py (tokenizer @ f19b3466);
longest prompt 48 tokens. Artifacts: the scene target " it" is identical across animals, so only the
description differs. The naming contexts are the exact description stems from those scene prompts, plus
3 neutral contexts; no new animal descriptions were written for P2. For P1, the split is by description
index across all animals: a held-out description and all 4 of its endings are excluded from training.
Filters: none beyond that exact stem extraction and the description-blocked split.

### Noise

12 scene prompts per animal; S2 allows two lost slots. Twelve capybara readouts are tested, so the random
baseline uses the 99th percentile rather than the 95th. Held-out margins on the cached data were thin at
the next token (0.004 to 0.08 rival-SD on the worst template), so a single prompt can flip a slot; S1 is
on the mean and S2 on slots for that reason. The scene prompts are clustered: only 3 descriptions per
animal, each repeated with 4 endings, so the 12 slots are not 12 independent descriptions. P1 respects
those clusters by cross-fitting on whole descriptions. P2 has only 3 capybara contexts, so its 2-of-3
rule is a validation gate, not a precise estimate of naming accuracy.

### Readout

Per readout: confirmed if V1 (word margin >= 1.0), V2 (tapir sum-4 set fails S1 and S2 for tapir), S1
(scene margin >= 1.0), S2 (>= 10/12 slots), S3 (> 99th percentile of random sets), C1 (ruby fraction
<= 0.35 at bara and next) and C2 (worst ruby prompt <= 0.50) all hold. Refuted (spelling-bound) only if
V1 and V2 hold, P1 and P2 both pass, and the scene margin is < 0.5 or slots are < 6/12. If a readout
would otherwise be refuted but P1 or P2 fails, it is inconclusive because the scene arm was not
validated. Reported as a South American mammal cluster if the top five are capybara, tapir, mara,
nutria and guinea pig. Inconclusive if V2 fails, nothing passes V1, every passing-V1 readout lands
between the confirmed and refuted bars, or a needed positive control fails. Any readout that neither
confirms nor refutes is inconclusive, including a best readout at scene margin 0.7 with 8 of 12 slots;
its exact margin and slots are reported. If the best readout is inconclusive, the PR headline is "no
confirmed capybara readout; best readout inconclusive," not a clean negative result. If refuted, the
PR's conclusion stands as is: no capybara neuron or group in these candidates.
