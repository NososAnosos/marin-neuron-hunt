"""Run with: python -m unittest scripts/test_check_claims.py"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_claims import explained_neuron, neurons_in_pr, plain_english_problems, problems  # noqa: E402

TEMPLATE = """## Neuron(s)

<!-- e.g. L40-N25546. Opening this PR as a draft claims the neuron(s) -->
{section}

## What's in this PR

See also L12-N3 for contrast.
"""


class NeuronsInPr(unittest.TestCase):
    def test_title_and_section_only(self):
        body = TEMPLATE.format(section="L40-N1, l41-n02")
        self.assertEqual(neurons_in_pr("Explain L7-N100", body), ["L7-N100", "L40-N1", "L41-N2"])

    def test_template_example_and_other_sections_do_not_count(self):
        self.assertEqual(neurons_in_pr("Tidy README", TEMPLATE.format(section="")), [])

    def test_missing_body_and_out_of_range(self):
        self.assertEqual(neurons_in_pr("L3-N4: both sides", None), ["L3-N4"])
        self.assertEqual(neurons_in_pr("L99-N1", None), [])


PLAIN = """## Neuron(s)

L40-N1

## In plain English

<!-- In a sentence or two per neuron, say what you believe it does. -->
{section}

## What's in this PR
"""


class PlainEnglish(unittest.TestCase):
    one = "It lights up on the word right before a list starts. The negative side is unclear."

    def test_missing_empty_or_comment_only_fails(self):
        self.assertEqual(len(plain_english_problems(["L40-N1"], TEMPLATE.format(section="L40-N1"))), 1)
        self.assertEqual(len(plain_english_problems(["L40-N1"], PLAIN.format(section=""))), 1)
        self.assertEqual(len(plain_english_problems(["L40-N1"], None)), 1)

    def test_too_short_fails(self):
        self.assertEqual(len(plain_english_problems(["L40-N1"], PLAIN.format(section="Fires on lists."))), 1)

    def test_one_neuron_needs_no_id(self):
        self.assertEqual(plain_english_problems(["L40-N1"], PLAIN.format(section=self.one)), [])

    def test_several_neurons_each_named(self):
        both = f"L40-N1: {self.one}\nL40-N2: {self.one}"
        self.assertEqual(plain_english_problems(["L40-N1", "L40-N2"], PLAIN.format(section=both)), [])
        errors = plain_english_problems(["L40-N1", "L40-N2"], PLAIN.format(section=f"L40-N1: {self.one} {self.one}"))
        self.assertEqual(len(errors), 1)
        self.assertIn("L40-N2", errors[0])


class ExplainedNeuron(unittest.TestCase):
    def test_paths(self):
        self.assertEqual(explained_neuron("explanations/L40/L40-N25546.yaml"), "L40-N25546")
        self.assertIsNone(explained_neuron("requests/L40-N25546-001.yaml"))


class Problems(unittest.TestCase):
    held = {"L40-N1": {"state": "claimed", "holder": {"github": "bo", "githubId": "2"}, "until": "2026-11-02T00:00:00Z"}}

    def test_someone_elses_claim_fails(self):
        errors = problems(["L40-N1"], ["L40-N1"], self.held, "1")
        self.assertEqual(len(errors), 1)
        self.assertIn("claimed by @bo, until 2026-11-02", errors[0])

    def test_own_claim_and_free_neurons_pass(self):
        self.assertEqual(problems(["L40-N1"], ["L40-N1"], self.held, "2"), [])
        self.assertEqual(problems(["L40-N9"], ["L40-N9"], self.held, "1"), [])

    def test_explained_neuron_with_no_holder_passes(self):
        claims = {"L40-N1": {"state": "explained", "holder": None, "explainedBy": "bo"}}
        self.assertEqual(problems(["L40-N1"], ["L40-N1"], claims, "1"), [])

    def test_undeclared_explanation_fails(self):
        errors = problems(["L40-N9"], [], {}, "1")
        self.assertEqual(len(errors), 1)
        self.assertIn("list it under '## Neuron(s)'", errors[0])


if __name__ == "__main__":
    unittest.main()
