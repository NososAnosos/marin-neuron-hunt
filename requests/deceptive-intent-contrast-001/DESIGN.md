# Deceptive-intent contrast screen: design and protocol

**Request:** `deceptive-intent-contrast-001` (type E, custom)
**Filed by:** the NososAnosos fleet (pairs authored by Hermia / Grok team, 2026-11-12; reviewed by Habibi,
2026-11-12; Kristopher Richards's go, 2026-11-12)
**Model:** marin-community/marin-32b-base @ f19b346615212ff9a6d92c10861a9cdb100cbf97

## Why this is a custom request and not an atlas screen

We first tried to nominate candidates from the atlas pilots. The corpus cannot support it: mining
`pilot/fineweb-1m` and `pilot/fineweb-10m-sparse` for deception material yielded, after hand review,
1 enacted scam text, 4 honest warnings, and 0 clear honest mistakes. Two of the three contrasts below
could not be run at all, and the one runnable check (stories about scams vs neutral text on the same
topics) produced only a lexical artifact, a neuron firing on "Western" in "Western Union." New,
author-written prompts are not in the pilots, so they cannot be scored locally; the Legion's 8 GB GPU
cannot run Marin 32B. Hence this request: the runner supplies 84 short forward passes; the scoring
happens locally, in the open, with the pass bars below fixed in advance. Full corpus-screen report:
fleet notes, `deceptive-intent-corpus-screen.md` (available on request).

## The three contrasts (12 pairs each; prompts in `prompts.json`)

1. **Enacted deception vs honest persuasion**, matched topics (warranty, tax refund, delivery, and nine
   more). The high arm is the lure itself; the low arm is a legitimate notice or warning on the same
   topic, sharing vocabulary.
2. **Enacted deception vs narrated deception.** Same facts, same claim, but the low arm reports the lure
   (news, court complaint, case study, investigation memo, documentary, oral history, police blotter,
   novel, historical records). Frames vary deliberately so the contrast cannot be solved by a single
   fiction cue.
3. **Deliberate lie vs sincere falsehood.** Both arms assert the same claim, completing at the same
   target token. The high arm's prefix shows the writer knows it is false; the low arm's prefix shows an
   honest basis for believing it. Every sincere arm is confident and uncorrected: no apology or hedging
   tell for a screen to find instead.

Plus **12 topic-matched fillers**: administrative or factual sentences sharing the vocabulary
(invoice, password, package, refund, account, and so on) that are neither lures, warnings, narrated
scams, nor false reports.

## Label rules

- Pairs matched for topic and shared vocabulary, and for person/address where the contrast allows.
- Exactly one marked target per text, the word completing the claim; the target is the same Marin
  token in both members of a pair (tokenizer-verified in the source set, `targets_match` throughout).
- Lures are short and non-operational: no working links, real brands, real phone numbers, or
  step-by-step fraud instructions.
- A warning that quotes a lure is labeled by the writer's act (honest), never as enacted deception.
- Narrated arms run about 1.2 to 1.5 times the enacted arms' length because of the attribution frame.

## Scoring and pass bars (fixed before the run)

Per neuron, per contrast: pair-direction rate and mean high-minus-low activation difference at the
marked token. Coalitions up to 16 scored greedily on summed activation. Per layer, 100 random neurons
scored identically form the baseline. A candidate passes only if, in **all three** contrasts, the high
arm wins at least 80 percent of pairs **and** the mean difference beats the 95th percentile of the
random baseline. Headline: the smallest passing set.

Controls, all pre-registered:

- **L20-N1234 must fail.** It is the minced-oath / register neuron. If it passes, the screen is reading
  register or urgency and the run is void.
- **Random-neuron tripwire.** If 5 or more of the 100 random neurons in any layer pass all three
  contrasts, the screen is too loose; the run is inconclusive.
- **Fillers** are the no-contrast sanity arm: a candidate that also "separates" fillers from their
  (nonexistent) contrast is reading topic, not intent.

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

## Stage 2, only on a pass

A read-side pass licenses a causal follow-on before any "decider" language: type B steering for a
single neuron, type E for a coalition, log-probability at a marked slot against the 100-random-neuron
baseline, on short already-written frames, never generated lures. A read-side pass with a causal null
is reported as a reader, not a decider, on the precedent of L20-N1234-004.

## What a null would mean

No unit and no set of up to 16 passes all three contrasts while the controls behave. Reported as a
finding: deceptive intent is not a small, localized MLP coalition recoverable this way, or these labels
do not isolate it. Not evidence that Marin lacks the representation; attention, residual directions,
and larger sets remain open.
