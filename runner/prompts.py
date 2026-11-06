"""Turns a request's [[ ]]-marked prompts into token ids and target positions. Needs only the tokenizer, so CI
can run the same check the GPU job does.
"""

import re
from dataclasses import dataclass

MODEL_REPO = "marin-community/marin-32b-base"
MODEL_SHA = "f19b346615212ff9a6d92c10861a9cdb100cbf97"
MAX_TOKENS = 512
MARKED = re.compile(r"^(.*?)\[\[(.+?)\]\](.*)$", re.S)


class PromptError(ValueError):
    pass


@dataclass
class Encoded:
    text: str
    ids: list
    pos: int  # index of the target token in ids (ids[0] is BOS)
    token: str


def load_tokenizer(path=None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path or MODEL_REPO, revision=None if path else MODEL_SHA)


def encode(tok, text):
    m = MARKED.match(text)
    if not m or "[[" in m.group(3):
        raise PromptError(f"{text!r}: mark exactly one target with [[ ]]")
    pre, target, post = m.groups()
    ids_pre = tok(pre)["input_ids"]
    ids_upto = tok(pre + target)["input_ids"]
    ids = tok(pre + target + post)["input_ids"]
    if ids_upto[:len(ids_pre)] != ids_pre or len(ids_upto) != len(ids_pre) + 1:
        n = len(tok(target, add_special_tokens=False)["input_ids"])
        raise PromptError(f"{text!r}: target {target!r} is {n} tokens here, not 1")
    if ids[:len(ids_upto)] != ids_upto:
        raise PromptError(f"{text!r}: the text after the target merges with it into another token")
    if len(ids) > MAX_TOKENS:
        raise PromptError(f"{text!r}: {len(ids)} tokens; the limit is {MAX_TOKENS}")
    if tok.bos_token_id is not None and ids[0] != tok.bos_token_id:
        raise PromptError(f"{text!r}: tokenizer didn't prepend BOS")
    return Encoded(text, ids, len(ids_upto) - 1, tok.convert_ids_to_tokens(ids_upto[-1]))


def single_token(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise PromptError(f"{text!r}: must be one token, is {len(ids)}")
    return ids[0]


def prompts_of(req):
    """Every marked prompt in a request, for the up-front check."""
    t = req.get("type")
    if t == "activation-test":
        return list(req["should_fire"]) + list(req["should_not_fire"])
    if t == "contrast":
        return [p[k] for p in req["pairs"] for k in ("high", "low")]
    if t in ("steering", "ablation"):
        return list(req["prompts"])
    return []


def single_tokens_of(req):
    t = req.get("type")
    if t == "steering":
        return list(req["expect_up"]) + list(req.get("expect_down") or [])
    if t == "ablation" and req["metric"]["kind"] == "next-token-logprob":
        return [req["metric"]["token"]]
    return []


def check(tok, req):
    """All problems with a request's prompts and named tokens; empty means it can run."""
    problems = []
    needs_tail = req.get("type") == "ablation" and req["metric"]["kind"] == "loss"
    for text in prompts_of(req):
        try:
            e = encode(tok, text)
            if needs_tail and e.pos == len(e.ids) - 1:
                problems.append(f"{text!r}: a loss metric needs text after the target")
        except PromptError as err:
            problems.append(str(err))
    for text in single_tokens_of(req):
        try:
            single_token(tok, text)
        except PromptError as err:
            problems.append(str(err))
    return problems
