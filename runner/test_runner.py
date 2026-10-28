"""Run with: python -m unittest runner/test_runner.py  (needs torch, transformers and the Marin tokenizer)"""

import json
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from jsonschema import Draft202012Validator

from runner import experiments, pick, report
from runner.model import Marin, Row, Stats
from runner.prompts import PromptError, check, encode, load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
TOK = load_tokenizer()


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    return Qwen3ForCausalLM(cfg).float().eval()


class FakeTok:
    pad_token_id = None
    eos_token_id = 1


class TestModelHooks(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.m = Marin(self.model, FakeTok(), stats=None)
        self.ids = [0, 5, 9, 12, 7, 3, 22]

    def recompute(self, ids, layer):
        got = {}
        mlp = self.model.model.layers[layer].mlp
        h = mlp.down_proj.register_forward_pre_hook(lambda mod, a: got.__setitem__("x", a[0].detach().clone()))
        self.model(input_ids=torch.tensor([ids]))
        h.remove()
        return got["x"][0]

    def test_read_matches_down_proj_input(self):
        acts, _, _ = self.m.run([Row(self.ids, 3)], reads=[(1, 7), (2, 40)])
        self.assertAlmostEqual(acts[0, 0], self.recompute(self.ids, 1)[3, 7].item(), places=5)
        self.assertAlmostEqual(acts[0, 1], self.recompute(self.ids, 2)[3, 40].item(), places=5)

    def test_early_stop_gives_same_reads(self):
        early, _, _ = self.m.run([Row(self.ids, 4)], reads=[(0, 3)])
        full, _, _ = self.m.run([Row(self.ids, 4)], reads=[(0, 3)], tokens=[5])
        self.assertAlmostEqual(early[0, 0], full[0, 0], places=6)

    def test_edit_only_affects_target_and_later(self):
        logits = self._logits_with_edit(3, 0, 10, 25.0)
        base = self._logits_with_edit(None, 0, 10, 0.0)
        self.assertTrue(torch.allclose(logits[:3], base[:3], atol=1e-6))
        self.assertFalse(torch.allclose(logits[3:], base[3:], atol=1e-3))

    def _logits_with_edit(self, pos, layer, neuron, value):
        def fn(mod, a):
            if pos is None:
                return None
            x = a[0].clone()
            x[0, pos, neuron] = value
            return (x,)
        h = self.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(fn)
        out = self.model(input_ids=torch.tensor([self.ids])).logits[0]
        h.remove()
        return out

    def test_edit_through_run_matches_manual(self):
        _, lp, _ = self.m.run([Row(self.ids, 3, [(0, 10, 25.0)])], tokens=[2, 9])
        manual = torch.log_softmax(self._logits_with_edit(3, 0, 10, 25.0)[3], -1)[[2, 9]]
        self.assertTrue(np.allclose(lp[0], manual.detach().numpy(), atol=1e-5))

    def test_reads_see_value_before_edit(self):
        clean, _, _ = self.m.run([Row(self.ids, 3)], reads=[(0, 10)])
        edited, _, _ = self.m.run([Row(self.ids, 3, [(0, 10, 25.0)])], reads=[(0, 10)])
        self.assertAlmostEqual(clean[0, 0], edited[0, 0], places=6)

    def test_padding_does_not_change_results(self):
        short = Row(self.ids[:4], 2, [(1, 5, 3.0)])
        long = Row(self.ids + [4, 4, 4], 6)
        alone_a, alone_lp, alone_loss = self.m.run([short], reads=[(2, 1)], tokens=[3], loss=True)
        both_a, both_lp, both_loss = self.m.run([long, short], reads=[(2, 1)], tokens=[3], loss=True)
        self.assertAlmostEqual(alone_a[0, 0], both_a[1, 0], places=5)
        self.assertAlmostEqual(alone_lp[0, 0], both_lp[1, 0], places=5)
        self.assertAlmostEqual(alone_loss[0], both_loss[1], places=5)


class FakeMarin:
    """Feeds preset activations / logprobs to the pass-rule code."""

    def __init__(self, acts=None, lps=None, losses=None, stats=None):
        self.tok, self.n_neurons = TOK, 200
        self._acts, self._lps, self._losses = acts, lps, losses
        self._stats = stats or {}
        self.rows = None

    def stats(self, layer, neuron):
        return self._stats.get((layer, neuron), dict(pos_q99=1.0, neg_q99=0.5, scale_abs_p999=2.0, mean=0.1))

    def run(self, rows, reads=(), tokens=None, loss=False):
        self.rows = rows
        acts = np.array(self._acts(rows) if callable(self._acts) else self._acts or np.zeros((len(rows), 1)))
        lps = self._lps(rows) if callable(self._lps) else self._lps
        losses = self._losses(rows) if callable(self._losses) else self._losses
        return acts.reshape(len(rows), -1), lps, losses


PROMPT = "The cat sat on the[[ mat]] today."


def act_req(n_fire=10, n_quiet=10, side="+", threshold="atlas-q99"):
    return dict(type="activation-test", neuron="L3-N7", side=side, hypothesis="h",
                should_fire=[PROMPT] * n_fire, should_not_fire=[PROMPT] * n_quiet, threshold=threshold,
                by=dict(github="x", method="human"))


class TestPassRules(unittest.TestCase):
    def test_activation_exactly_80_percent_passes(self):
        acts = [[2.0]] * 8 + [[0.5]] * 2 + [[0.5]] * 8 + [[2.0]] * 2
        res = experiments.activation_test(FakeMarin(acts), act_req(), "L3-N7-001")
        self.assertEqual(res["outcome"], "confirmed")
        self.assertEqual(res["summary"]["should_fire_above"], 0.8)

    def test_activation_70_percent_refuted(self):
        acts = [[2.0]] * 7 + [[0.5]] * 3 + [[0.5]] * 10
        self.assertEqual(experiments.activation_test(FakeMarin(acts), act_req(), "x")["outcome"], "refuted")

    def test_activation_negative_side_uses_magnitude(self):
        # neg_q99 = 0.5 is a magnitude; -0.9 is past it, -0.1 is not
        acts = [[-0.9]] * 10 + [[-0.1]] * 10
        res = experiments.activation_test(FakeMarin(acts), act_req(side="-"), "x")
        self.assertEqual(res["outcome"], "confirmed")
        self.assertEqual(res["summary"]["threshold"], 0.5)
        self.assertEqual(res["summary"]["auroc"], 1.0)

    def test_activation_numeric_threshold(self):
        acts = [[0.6]] * 10 + [[0.3]] * 10
        res = experiments.activation_test(FakeMarin(acts), act_req(threshold=0.5), "x")
        self.assertEqual((res["outcome"], res["summary"]["threshold"]), ("confirmed", 0.5))

    def test_contrast(self):
        req = dict(type="contrast", neuron="L3-N7", side="+", pairs=[dict(high=PROMPT, low=PROMPT)] * 10)
        acts = [[1.0], [0.2]] * 10
        self.assertEqual(experiments.contrast(FakeMarin(acts), req, "x")["outcome"], "confirmed")
        acts = [[1.0], [0.8]] * 10  # right direction, mean diff 0.2 < 0.5 * pos_q99
        self.assertEqual(experiments.contrast(FakeMarin(acts), req, "x")["outcome"], "refuted")

    def steering_lps(self, target_gain, baseline_gain):
        def lps(rows):
            out = np.zeros((len(rows), 1))
            for i, r in enumerate(rows):
                if r.edits:
                    out[i] = target_gain if r.edits[0][1] == 7 else baseline_gain
            return out
        return lps

    def test_steering_beats_baseline(self):
        req = dict(type="steering", neuron="L3-N7", side="+", k=4, prompts=[PROMPT] * 10, expect_up=[" vice"])
        m = FakeMarin(lps=self.steering_lps(1.0, 0.1))
        res = experiments.steering(m, req, "L3-N7-001")
        self.assertEqual(res["outcome"], "confirmed")
        target_edit = m.rows[10].edits[0]
        self.assertEqual(target_edit, (3, 7, 8.0))  # +4 x scale_abs_p999 (2.0)
        self.assertEqual(len(res["baseline_neurons"]), 100)
        self.assertNotIn(7, res["baseline_neurons"])
        res = experiments.steering(FakeMarin(lps=self.steering_lps(0.05, 0.1)), req, "L3-N7-001")
        self.assertEqual(res["outcome"], "refuted")

    def test_steering_baseline_is_deterministic(self):
        req = dict(type="steering", neuron="L3-N7", side="-", prompts=[PROMPT] * 10, expect_up=[" vice"])
        a = experiments.steering(FakeMarin(lps=self.steering_lps(1, 0)), req, "L3-N7-002")["baseline_neurons"]
        b = experiments.steering(FakeMarin(lps=self.steering_lps(1, 0)), req, "L3-N7-002")["baseline_neurons"]
        self.assertEqual(a, b)

    def ablation_req(self, predicted):
        return dict(type="ablation", neurons=["L3-N7", "L5-N1"], mode="mean", prompts=[PROMPT] * 10,
                    metric=dict(kind="next-token-logprob", token=" vice"), predicted_effect=predicted)

    def ablation_lps(self, target_change, baseline_change):
        def lps(rows):
            out = np.zeros((len(rows), 1))
            for i, r in enumerate(rows):
                if r.edits:
                    out[i] = target_change if (3, 7, 0.1) in r.edits else baseline_change
            return out
        return lps

    def test_ablation_outcomes(self):
        cases = [(-1.0, -0.9, "confirmed"), (-1.0, -0.3, "inconclusive"), (-1.0, 0.5, "refuted"),
                 (-1.0, 0.0, "inconclusive")]
        for predicted, change, want in cases:
            res = experiments.ablation(FakeMarin(lps=self.ablation_lps(change, 0.0)), self.ablation_req(predicted), "x")
            self.assertEqual(res["outcome"], want, (predicted, change))

    def test_ablation_random_sets_match_layers(self):
        m = FakeMarin(lps=self.ablation_lps(-1, 0))
        res = experiments.ablation(m, self.ablation_req(-1.0), "x")
        for s in res["baseline_sets"]:
            self.assertEqual([n.split("-")[0] for n in s], ["L3", "L5"])
            self.assertNotIn("L3-N7", s)


class TestPrompts(unittest.TestCase):
    def test_target_position_and_bos(self):
        e = encode(TOK, "Her handwriting was bold[[ and]] impossible to ignore.")
        self.assertEqual((e.ids[0], e.token, e.ids[e.pos]), (TOK.bos_token_id, "Ġand", TOK(" and", add_special_tokens=False)["input_ids"][0]))

    def test_multi_token_target_rejected(self):
        with self.assertRaises(PromptError):
            encode(TOK, "The cluster runs[[ Kubernetesxyzzy]] on bare metal.")

    def test_unmarked_and_double_marked_rejected(self):
        for text in ("no marker here", "two[[ a]] and[[ b]]"):
            with self.assertRaises(PromptError):
                encode(TOK, text)

    def test_check_collects_problems(self):
        req = act_req()
        req["should_fire"] = req["should_fire"][:9] + ["bad[[ zzqqxxyy]] one"]
        self.assertEqual(len(check(TOK, req)), 1)
        self.assertEqual(check(TOK, act_req()), [])

    def test_loss_metric_needs_text_after_target(self):
        req = dict(type="ablation", neurons=["L3-N7"], prompts=["Ends on the[[ mat]]"] * 10,
                   metric=dict(kind="loss"), predicted_effect=1.0)
        self.assertEqual(len(check(TOK, req)), 10)


class TestPick(unittest.TestCase):
    def item(self, i, req):
        return pick.Pending(f"L1-N{i}-001", Path("x"), req, i, "c")

    def test_cap_keeps_merge_order(self):
        small = act_req()
        big = dict(type="steering", prompts=[PROMPT] * 200)
        items = [self.item(1, small), self.item(2, big), self.item(3, small)]
        cap = pick.est_cost([items[0]]) + 0.01
        run, custom, later = pick.batch(items, cap)
        self.assertEqual([i.id for i in run], ["L1-N1-001"])
        # the small request after an over-cap one waits too, so merge order holds
        self.assertEqual([i.id for i in later], ["L1-N2-001", "L1-N3-001"])

    def test_custom_goes_to_maintainer(self):
        run, custom, later = pick.batch([self.item(1, dict(type="custom"))], 10)
        self.assertEqual((len(run), len(custom), len(later)), (0, 1, 0))


class TestReport(unittest.TestCase):
    def test_record_validates_and_comment_renders(self):
        res = experiments.activation_test(FakeMarin([[2.0]] * 10 + [[0.1]] * 10), act_req(), "L3-N7-001")
        res["seconds"] = 1.0
        item = dict(id="L3-N7-001", request=act_req(), commit="abc1234def")
        rec = report.record(item, res, "f00dfeed", "2026-10-28", "NVIDIA H200")
        schema = json.loads((ROOT / "schema" / "result.schema.json").read_text())
        errors = list(Draft202012Validator(schema).iter_errors(yaml.safe_load(report.dump(rec))))
        self.assertEqual(errors, [])
        text = report.comment(rec)
        self.assertIn("**confirmed**", text)
        self.assertIn("`Ġmat`", text)


class TestStats(unittest.TestCase):
    def test_lookup(self):
        import pyarrow as pa

        t = pa.table(dict(layer=[0, 0, 2], neuron=[1, 5, 1], pos_q99=[0.1, 0.5, 0.9]))
        s = Stats(t, n_neurons=8)
        self.assertEqual(s(0, 5)["pos_q99"], 0.5)
        self.assertEqual(s(2, 1)["pos_q99"], 0.9)
        with self.assertRaises(KeyError):
            s(1, 1)


if __name__ == "__main__":
    unittest.main()
