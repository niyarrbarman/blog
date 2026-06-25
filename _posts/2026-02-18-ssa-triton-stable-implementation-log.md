---
layout: post
title: "How I Stabilized SSA in Triton"
date: 2026-02-18 00:00:00 +0000
categories: [learning]
tags: [ssa, triton, nemo, nemotron, attention, kernels]
author: niyar r barman
excerpt: "i implemented SSA in triton on a 114m nemotron variant and stabilized training end-to-end."
---

i spent the last few weeks implementing SSA end-to-end and it was messy.

## quick paper context

SSA comes from [Scaled Signed Averaging Improves In-Context and Early Learning Benchmark Performance in Small Transformers](https://arxiv.org/pdf/2508.14685). the core idea is to replace plain softmax scoring with a signed, parameterized transform (`n` and `b`) that is less brittle in practice. the paper reports better in-context and early-learning behavior on small transformer settings, which is exactly why i wanted to try it in my stack.

the original SSA weight expression is the `1 + b|x|` form:

$$
w_i = (1 + b|x_i|)^{n \,\mathrm{sgn}(x_i)}.
$$

in kernels, i mostly used the mathematically equivalent log-exp form:

$$
w_i = \exp\left(n\,\mathrm{sgn}(x_i)\,\ln(1 + b|x_i|)\right).
$$

so normalized SSA can be written as:

$$
p_i = \frac{w_i}{\sum_j w_j}
= \mathrm{softmax}(z_i), \quad
z_i = n\,\mathrm{sgn}(x_i)\,\ln(1+b|x_i|).
$$

that equivalence matters because some Triton builds were happier with `exp(log(.))` style math than direct `pow`.

## model i used (nemotron 4b -> ~114m)

i started from nvidia's nemotron3_4b recipe and shrank it to a much smaller ~114m model so i could iterate fast:

```python
from nemo.collections.llm.recipes.nemotron3_4b import (
    pretrain_recipe as pretrain_base_recipe,
)


def pretrain_recipe(**kwargs):
    recipe = pretrain_base_recipe(**kwargs)
    
    # Model architecture
    recipe.model.config.num_layers = 12
    recipe.model.config.num_attention_heads = 24
    recipe.model.config.num_query_groups = 8
    recipe.model.config.hidden_size = 768
    recipe.model.config.ffn_hidden_size = 3072
    recipe.model.config.kv_channels = None
    recipe.model.config.share_embeddings_and_output_weights = True
    
    # Parallelism
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.strategy.tensor_model_parallel_size = 1
    
    return recipe
```

this one decision saved me a lot of time while debugging kernels.

## what i implemented

### 1) unfused SSA baseline first (reference truth)

before any fusion work, i kept a clean reference path (`ssa_attention.py`) with canonical SSA behavior:

- transform: `n * sign(x) * log1p(b * |x|)`
- normalize row-wise
- apply dropout on attention probabilities before `@V`

that became the correctness anchor for parity checks.

baseline transform:

```python
def ssa_transform(self, x: torch.Tensor) -> torch.Tensor:
    n, b = self.get_ssa_params()
    abs_x = torch.abs(x)
    sign_x = torch.sign(x)
    log_term = torch.log1p(b * abs_x)
    return n * sign_x * log_term
```

baseline dropout placement (important for parity):

```python
# Apply SSA softmax
attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

# Dropout on probabilities (before @V)
attention_probs = self.attention_dropout(attention_probs)

# Compute context
context = torch.bmm(attention_probs, value.transpose(0, 1))
```

### 2) flexattention attempt (and why i moved on)

i tried flexattention first for performance, but i ran into three recurring issues:

- trainable SSA params were accidentally detached to python floats in early score_mod logic
- backend `AUTO` kernel-option path produced runtime/compiler friction
- `torch.compile` + learnable score_mod captures made backward unstable in my setup

early trap in flex path:

```python
# This made n/b non-learnable through score_mod
n_val = float(n.detach())
b_val = float(b.detach())
```

i patched backend handling so `AUTO` did not leak bad kernel options:

```python
self._flex_backend = backend
self._flex_kernel_options = None if backend == "AUTO" else {"BACKEND": backend}
```

and i added compile gating when learnable score_mod was active:

```python
if self._use_torch_compile and self.learnable_ssa:
    self._use_torch_compile = False
```

this made it more usable, but still costly and fragile for long iteration loops.

### 3) original triton path: speed win, quality loss

the original fused triton kernels were significantly faster, but long-run training diverged compared to baseline. early losses looked okay, then the gap widened as steps increased. this was the point where i stopped trusting "it runs" and started treating this as a math+integration parity problem.

key fixes in this phase:

- align forward/backward with original SSA normalization semantics
- fix `db` gradient sign term
- add compensated accumulation for `dn`/`db`
- remove `tl.pow` dependency for better triton compatibility

the `db` sign issue is easier to see from derivatives.
if

$$
z_i = n\,\mathrm{sgn}(x_i)\,\ln(1+b|x_i|),
$$

then

$$
\frac{\partial z_i}{\partial n} = \mathrm{sgn}(x_i)\,\ln(1+b|x_i|),
\quad
\frac{\partial z_i}{\partial b} =
n\,\mathrm{sgn}(x_i)\,\frac{|x_i|}{1+b|x_i|}.
$$

that `sgn(x_i)` factor also belongs in the `db` path.

kernel-side fix looked like this:

```python
# old (wrong sign behavior)
db_acc += tl.sum(ds_ssa * ssa_n * abs_s / denom)

# corrected
db_acc += tl.sum(ds_ssa * ssa_n * sign_s * abs_s / denom)
```

i also replaced `tl.pow` with equivalent log-exp math for compatibility:

```python
log_one_plus_bs = tl.log(one_plus_bs)
ssa_w = tl.where(valid, tl.exp(ssa_exp * log_one_plus_bs), 0.0)
```

these fixes helped, but this branch still was not the final stable path.

### 4) triton stable reboot

after spending another week, i moved to a tutorial-structured triton stable implementation:

- stage-based causal handling
- split backward kernels (`dQ` and `dKV`)
- cleaner normalization state handling
- explicit GQA-safe indexing (`z`, `h_q`, `h_kv`)

this was the actual turning point. instead of patching older assumptions, i reset the architecture around a known-good fused-attention structure and integrated SSA into that.


one important stabilization was moving to tutorial-style stored normalization state (`M`) and matching backward recomputation to it:

```python
# forward stores m = m_i + log2(l_i_safe)
m = m_i + tl.math.log2(l_i_safe)
tl.store(l_ptrs, m, mask=l_mask)
```

```python
# backward reconstructs p from base-2 path and stored m_i
t2 = tl.where(valid, ssa_n * sign_s * log_opbs * RCP_LN2, NEG_LARGE)
p = tl.where(valid, tl.math.exp2(t2 - m_i[:, None]), 0.0)
```

and i made head/batch indexing explicit instead of flattened assumptions:

```python
off_z = off_hz // H_Q
off_h_q = off_hz % H_Q
off_h_kv = off_h_q // GQA_RATIO
```

### 5) final stability lever: force contiguous q/k/v

the last quality fix was simple and practical: materialize q/k/v as contiguous tensors before the triton call. once i made `force_contiguous_qkv` default in the training path, behavior became much more stable in the strided layouts i actually care about.

module hook:

```python
if self.force_contiguous_qkv:
    query_t = query_t.contiguous()
    key_t = key_t.contiguous()
    value_t = value_t.contiguous()
```

launcher flag:

```python
parser.add_argument(
    "--force_contiguous_qkv",
    action="store_true",
    default=False,
    help="Materialize Q/K/V as contiguous tensors before Triton attention call.",
)
```

script default:

```bash
FORCE_CONTIGUOUS_QKV=${FORCE_CONTIGUOUS_QKV:-1}
```

## final training policy i pinned

to avoid silent config drift while comparing runs, i pinned:

- kernel variant: `stable` only
- `n` learnable, initialized at `1.5`
- `b` fixed at `0.8`
- warmup with RNG snapshot/restore for reproducibility
- contiguous q/k/v enabled by default

policy enforcement in launcher:

```python
if args.learnable_b:
    logger.warning("Ignoring --learnable_b; b is fixed by policy.")
if args.ssa_n != 1.5 or args.ssa_b != 0.8:
    logger.warning(
        "Overriding SSA init from (n=%s, b=%s) to fixed policy (n=1.5, b=0.8).",
        args.ssa_n,
        args.ssa_b,
    )
args.learnable_b = False
args.ssa_n = 1.5
args.ssa_b = 0.8
```

## final stable training loss

final loss curve merged into one continuous line over global steps `0 -> 29999`.

<div id="ssa-final-loss-plot" style="width: 100%; height: 460px;"></div>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
  (function () {
    if (typeof Plotly === "undefined") return;

    const target = document.getElementById("ssa-final-loss-plot");
    if (!target) return;
    const root = document.documentElement;

    const dataUrl =
      "{{ '/assets/data/loss_vs_step_tr_bbyluc_ssa_triton_77025_77104_77235.json' | relative_url }}";
    const themePalette = {
      light: {
        bg: "#fdfbf7",
        text: "#3d3632",
        grid: "#e0d8cc",
        line: "#b91c1c",
      },
      dark: {
        bg: "#1a1814",
        text: "#d4cdc0",
        grid: "#3d362c",
        line: "#facc15",
      },
    };

    function getThemeName() {
      return root.getAttribute("data-theme") === "dark" ? "dark" : "light";
    }

    function getThemeColors() {
      return themePalette[getThemeName()];
    }

    fetch(dataUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then((payload) => {
        const entries = Object.entries(payload.series || {});
        if (!entries.length) {
          throw new Error("No series found in JSON.");
        }

        entries.sort(
          (a, b) => (a[1].min_step || 0) - (b[1].min_step || 0)
        );

        const merged = [];

        entries.forEach(([, series]) => {
          const pairs = Array.isArray(series.step_loss) ? series.step_loss : [];

          pairs.forEach((pair) => {
            if (Array.isArray(pair) && pair.length >= 2) {
              merged.push(pair);
            }
          });
        });

        merged.sort((a, b) => a[0] - b[0]);
        const mergedX = merged.map((p) => p[0]);
        const mergedY = merged.map((p) => p[1]);
        const c = getThemeColors();
        const trace = {
          x: mergedX,
          y: mergedY,
          mode: "lines",
          name: "SSA Triton Stable",
          line: { width: 2.5, color: c.line },
        };

        const layout = {
          title: { text: "SSA Triton Stable: Training Loss vs Step", x: 0.02 },
          xaxis: { title: "Global step", gridcolor: c.grid },
          yaxis: { title: "Loss", gridcolor: c.grid },
          hovermode: "x unified",
          margin: { l: 65, r: 20, t: 55, b: 55 },
          font: { color: c.text },
          plot_bgcolor: c.bg,
          paper_bgcolor: c.bg,
          showlegend: false,
        };

        const config = { responsive: true, displaylogo: false };
        Plotly.newPlot(target, [trace], layout, config);

        const observer = new MutationObserver(() => {
          const t = getThemeColors();
          Plotly.restyle(target, { "line.color": t.line }, [0]);
          Plotly.relayout(target, {
            plot_bgcolor: t.bg,
            paper_bgcolor: t.bg,
            "xaxis.gridcolor": t.grid,
            "yaxis.gridcolor": t.grid,
            "font.color": t.text,
          });
        });
        observer.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
      })
      .catch((err) => {
        target.textContent = "failed to load loss json: " + err.message;
      });
  })();
</script>

## what i learned

- kernel speedups are easy to get; long-horizon training parity is the real test
- tiny semantic mismatches (normalization state, dropout placement, gradient terms) are enough to derail learning
- backend/compiler ergonomics can dominate iteration speed
- layout assumptions are not cosmetic; contiguity can be the difference between "works" and "trains"
- keeping one clean unfused reference implementation is mandatory when doing fused kernel work

more detailed commit-by-commit notes live in my internal timeline doc, but this is the practical story of how the implementation actually converged.

## refs

1. Naim O, Bhar S, Bolte J, & Asher N (2025). *[Scaled Signed Averaging Improves In-Context and Early Learning Benchmark Performance in Small Transformers](https://arxiv.org/pdf/2508.14685)*. 
2. OpenAI kernel team. (n.d.). *[Fused Attention (Triton Tutorial 06)](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)*. Triton documentation.
