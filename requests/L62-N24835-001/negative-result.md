# Capybara hunt: negative result

Request: `L62-N24835-001` (scene control, type E). This note reports the hunt behind it. Data: `probes/capybara-v1` on `Station-house/marin-32b-neuron-atlas` @ `6402b9e`, cached dense MLP activations (`act = silu(gate) * up`) for all 64 layers. CPU only, no new forward passes. Screened 2026-11-10.

## Result

No clean capybara neuron, and no clean group of up to 16 neuron-sides. Nothing passes every bar at once: a margin over the nearest animals (tapir, mara, guinea pig, nutria) at the completing token and the next token, firing on concept and other-language prompts, and staying quiet on the Ruby testing gem and on bare `bara`. The best groups separate capybara only when the word is spelled out, they carry Ruby-gem signal, and the same search finds equally clean groups for a fake tapir target.

The ten animals are capybara, guinea pig, beaver, nutria, mara, tapir, hippo, otter, wombat and dog. " capybara" is three Marin tokens (` cap`, `y`, `bara`), so the completing-token site is `bara`.

## 1. Single neurons

Start: `summaries/pair_screen.parquet`, the top 200 neurons per layer, site (`word_last`, `next`) and pole by Cohen's d of capybara against the mean of the other nine animals. A side was kept if it was in the top 200 at both sites with the same sign, concept and multilingual prompts moved at least 40% as far as the capybara word, and the Ruby gem and bare `bara` moved at most 35% as far. 39 sides survived. They were re-ranked from the dense activations by capybara's margin over the strongest single rival, which a nine-animal mean can hide.

Ranked candidates, none clean:

1. **`L62-N24835+`**: best positive-side gap at `bara` (capybara 0.62, tapir 0.41, beaver 0.29; nutria -1.10, guinea pig -0.53). Other-language names fire (chigüiro 2.69, carpincho 2.05). But the next token fires for every animal (tapir 1.87, capybara 1.85), and 0.62 is far below this side's atlas `pos_q99` of 9.47 on `pilot/fineweb-1m`. Its atlas tail is unrelated: the negative side fires just before generic food words.
2. **`L63-N27150-`**: largest gap (capybara -6.44), but tapir -5.48, guinea pig -4.45, mara -4.42, nutria -3.73. A South American mammal cluster, not a capybara neuron.
3. **`L57-N14263+`**: small and cleaner at the word (capybara 0.24, guinea pig 0.15, nutria 0.13, mara 0.12), with a next-token margin over mara of only 0.01.
4. **`L46-N25555+`**: concept-last (0.45) is above the capybara word (0.19); mara is close behind (0.13).

`L54-N2754-` ties nutria and tapir with capybara and is not ranked. The raw top of the screen is layer-0 units with d around 20 to 30 that fire on the Ruby gem and bare `bara`, not on the concept.

## 2. Groups of up to 10 sides

- **Pool:** 1,301 capybara sides: the 39 above, plus the top sides per layer and pole by d at either or both sites.
- **Search:** greedy forward selection with two readouts, a standardized sum and an L2 logistic probe, scored on 10 random 10/10 splits of the 20 templates.
- **Margin:** capybara's held-out mean minus the strongest rival's, divided by the SD of the rival scores ("rival-SD units").
- **Caps:** the single-neuron caps were enforced on the group score at `bara`.
- **Honest check:** an outer 2-fold check re-selected each set on one half of the templates and tested it on the other.

**Closest candidate:** the 4-side sum `L16-N22982+ L23-N7365+ L47-N12076+ L9-N4603-`. Its held-out margin is 3.37 at the word and 2.73 at the next token, and the caps pass on means (Ruby 0.32, bare `bara` -0.01, concept 0.41, multilingual 0.42). It fails because:

- ruby_gem08 ("The capybara-webkit driver is deprecated.") scores 1.10 of the capybara signal at `bara`. Up to that token the text is "The capybara", which is identical to an animal sentence, so no readout at that position can pass it.
- At the next token the mean Ruby share is 0.82.
- The `#capybara` template (tokenized `c apy bara`) is lost in every split where it is held out.

**Controls:**

- **Tapir fake target:** the same search wins every held-out template from size 4 (sum), mostly with layer-0 units. Capybara with the caps off does it at size 3 with a layer-0 `bara` unit.
- **Random sets** (300 per size) are far below: all-layer random sets have a 95th percentile of at most 0.08 and a best of at most 0.53. The search works; the problem is specificity.

## 3. Extending to 16

Nothing material changes. Margins rise by 0.07 to 0.42 per step.

- **Thin wins:** on the outer fold without `#capybara`, the capped sum sets at 12 and 16 and the capped logistic set at 16 win every held-out template. The worst template wins by only 0.004 to 0.08 rival-SD at the next token. The logistic set at 12 wins 100% at the word and 90% at the next token.
- **Ruby gem:** in those outer tests ruby_gem08 scores 1.03 to 1.27. On that fold the next-token Ruby share is 0.46 to 0.66, and 0.56 to 0.79 across the 10 splits, against a bar of 0.35.
- **Tapir:** on honest held-out evidence capybara and tapir are indistinguishable. Tapir wins every template on the other fold at 12 and 16, while capybara still loses `#capybara`.
- **Membership:** the groups sit in early and middle layers and lean on `y`, `bara` and the next token, not on ` cap`. Of the 39 filtered sides, only `L50-N4018-` ever enters a capped set (logistic, step 12). None of the four ranked singles enters any run. No side from layers 51 to 63 enters a capped set (`L51-N3149-` enters the uncapped run at step 8).

## 4. Why the scene control

At later tokens of the pair sentences (109 matched items), `L62-N24835+` is not capybara-specific: capybara minus tapir is -0.40 (t = -3.6). The capped groups do score capybara above tapir there (logistic size 10: 97% of items). But those tokens come from the sentences the readouts were fit on, so they may carry the earlier `cap y bara` text forward. Separating the animal from its spelling needs prompts that describe a capybara without naming it, which means new forward passes on Marin 32B. Request `L62-N24835-001` asks for that. Our prior is that it will come back refuted.

## Caveats

- Only 20 pair templates; a held-out half has 10, so a 90% win rate is one lost template.
- The group pool was ranked on all 20 templates, which leaks a little into the held-out splits. The outer 2-fold numbers are the honest ones.
- Margins in rival-SD units are not comparable to single-side activation gaps.
- The analysis scripts and CSVs are not in this PR; they can be shared on request.
