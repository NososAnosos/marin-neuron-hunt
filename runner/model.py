"""Marin 32B with read/write access to MLP neurons, plus the atlas's per-neuron stats.

A neuron's activation is the atlas's: act = silu(gate) * up, the input to mlp.down_proj. Reads and edits happen
at one position per prompt (the [[ ]] target). Reads see the value before any edit at that layer.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

BATCH = 16


@dataclass
class Row:
    ids: list
    pos: int
    edits: list = field(default_factory=list)  # [(layer, neuron, value)] applied at pos


class _Stop(Exception):
    pass


class Stats:
    """pole_stats.parquet, indexed by (layer, neuron)."""

    def __init__(self, table, n_neurons):
        self.n = n_neurons
        cols = table.column_names
        layer = table["layer"].to_numpy().astype(np.int64)
        neuron = table["neuron"].to_numpy().astype(np.int64)
        self.index = np.full(int(layer.max() + 1) * n_neurons, -1, dtype=np.int64)
        self.index[layer * n_neurons + neuron] = np.arange(len(layer))
        self.cols = {c: table[c].to_numpy() for c in cols if c not in ("layer", "neuron")}

    def __call__(self, layer, neuron):
        i = self.index[layer * self.n + neuron] if layer * self.n + neuron < len(self.index) else -1
        if i < 0:
            raise KeyError(f"no atlas stats for L{layer}-N{neuron}")
        return {c: float(v[i]) for c, v in self.cols.items()}


class Marin:
    def __init__(self, model, tok, stats):
        self.model, self.tok, self.stats = model.eval(), tok, stats
        self.layers = model.model.layers
        self.n_neurons = self.layers[0].mlp.down_proj.in_features
        self.device = next(model.parameters()).device

    @classmethod
    def load(cls, model_dir, pole_stats):
        import pyarrow.parquet as pq
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                     device_map="cuda")
        n = model.model.layers[0].mlp.down_proj.in_features
        return cls(model, AutoTokenizer.from_pretrained(model_dir), Stats(pq.read_table(pole_stats), n))

    @torch.no_grad()
    def run(self, rows, reads=(), tokens=None, loss=False):
        """Forward every row; returns (acts [R, len(reads)], token logprobs at pos [R, len(tokens)] or None,
        mean next-token loss over the predictions from pos onward [R] or None)."""
        acts = np.zeros((len(rows), len(reads)), dtype=np.float64)
        lps = np.zeros((len(rows), len(tokens))) if tokens else None
        losses = np.zeros(len(rows)) if loss else None
        need_logits = bool(tokens) or loss
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            a, lp, ls = self._batch(chunk, reads, tokens, loss, need_logits)
            acts[start:start + len(chunk)] = a
            if lps is not None:
                lps[start:start + len(chunk)] = lp
            if losses is not None:
                losses[start:start + len(chunk)] = ls
        return acts, lps, losses

    def _batch(self, rows, reads, tokens, loss, need_logits):
        width = max(len(r.ids) for r in rows)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for b, r in enumerate(rows):
            ids[b, :len(r.ids)] = torch.tensor(r.ids)
            mask[b, :len(r.ids)] = 1
        pos = torch.tensor([r.pos for r in rows])
        rows_idx = torch.arange(len(rows))
        acts = torch.zeros((len(rows), len(reads)))

        by_layer = {}
        for j, (layer, neuron) in enumerate(reads):
            by_layer.setdefault(layer, {"reads": [], "edits": []})["reads"].append((j, neuron))
        for b, r in enumerate(rows):
            for layer, neuron, value in r.edits:
                by_layer.setdefault(layer, {"reads": [], "edits": []})["edits"].append((b, neuron, value))
        last = max(by_layer) if by_layer else -1

        def hook(layer, spec):
            def fn(module, args):
                x = args[0]
                for j, neuron in spec["reads"]:
                    acts[:, j] = x[rows_idx, pos, neuron].float().cpu()
                if spec["edits"]:
                    x = x.clone()
                    for b, neuron, value in spec["edits"]:
                        x[b, pos[b], neuron] = value
                if not need_logits and layer == last:
                    raise _Stop
                return (x,) + tuple(args[1:])
            return fn

        handles = [self.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l, s)) for l, s in by_layer.items()]
        try:
            out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device))
        except _Stop:
            out = None
        finally:
            for h in handles:
                h.remove()
        lp = ls = None
        if need_logits:
            logits = out.logits
            if tokens:
                at = torch.log_softmax(logits[rows_idx, pos].float(), dim=-1)
                lp = at[:, tokens].cpu().numpy()
            if loss:
                ls = np.zeros(len(rows))
                for b, r in enumerate(rows):
                    pred = logits[b, r.pos:len(r.ids) - 1].float()
                    gold = ids[b, r.pos + 1:len(r.ids)].to(pred.device)
                    ls[b] = torch.nn.functional.cross_entropy(pred, gold).item() if len(gold) else np.nan
        return acts.numpy(), lp, ls
