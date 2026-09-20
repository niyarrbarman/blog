---
layout: post
title: "Specializing and Distilling a Model from 23B to 8B"
date: 2026-09-21 00:00:00 +0200
categories: [learning]
tags: [distillation, pruning, forward-kl, megatron-lm, luciole, llms, nvidia, minitron,]
author: niyar r barman
excerpt: "i specialised a 23b teacher for math and pruned it twice."
plotly: true
---

this was the main story of my masters internship: can i take a 23b open model, make it better at math, and then turn it into something much smaller without throwing that work away?

the short answer is mostly yes. i ended with an 8.36b model that kept 99.2% of its teacher's mean mathematical accuracy while using 36% of its parameters. the less convenient answer is that this did not come for free: general knowledge dropped too, and one of my early assumptions about pruning turned out to be wrong.

## data🥀

before i could trust a training run, i had to trust the data that was entering it. the first failed continued-pretraining run made this very concrete: source labels and aggregate weights were not enough. two token-level things mattered a lot more than i expected: whether packed sequences ever showed the model an end-of-sequence token, and whether supposedly different documents were exact duplicates.

the sequence-length problem was especially troublesome. with 4k-token windows, 74.1% of the blend's tokens lived in documents longer than a window, and 51.8% of packed windows had no EOS token at all. that is not a great setup for teaching a generative model when to stop. moving to 32k windows brought the blend-wide no-EOS rate down to 7.3%.

<div id="window-length-chart" class="plotly-figure" aria-label="Effect of training sequence length on documents longer than a window and missing end-of-sequence tokens"></div>

the 32k change did not solve every source. `math_proofs_v2` documents averaged roughly 65k tokens, so 52% of its windows still had no EOS signal. no reasonable context length would repair that without changing how the documents were segmented, so i dropped it. arXiv was less pathological: at 32k it was at 10%, so i kept it but discarded documents beyond the cap, reducing it from 4.50b to 3.11b tokens.

<div id="source-eos-chart" class="plotly-figure plotly-figure-compact" aria-label="End-of-sequence rates for the cleaned corpus and its problematic sources"></div>

the other problem was duplication. a global hash pass found that 47% of `math_v2` documents were exact duplicates. a preliminary sample had hidden most of that. after the 32k length policy and global exact deduplication, the pool went from 179.03b to 112b tokens across 145 shards: 69% retained. the final sampler used those cleaned token counts directly - no manual bucket multiplier - which gave a maths-heavy but less lopsided mix.

<div id="data-mix-chart" class="plotly-figure plotly-figure-compact" aria-label="Semantic composition of the cleaned 112B-token training corpus"></div>

## the setup

i started with Luciole-23B, a French-oriented base model. before compressing anything, i continued pretraining it for 50b tokens on a cleaned, maths-heavy mix. this gave me a better teacher to compress rather than just a smaller copy of the original model.

the maths average across five benchmarks went from 42.51 to 50.17. the largest gains were on the harder tasks: MINERVA-MATH gained 7.68 points and MMLU-Pro maths gained 8.29 points. MMLU also went from 59.02 to 61.59, which was a useful sanity check that the teacher correction was not simply deleting general knowledge.

then i compressed it in two steps: 23.22b to 13.98b, then 13.98b to 8.36b. each cut uses structured pruning to remove whole layers and width components, followed by a full distillation run to make the new architecture usable again.

<button class="image-lightbox-trigger" type="button" data-lightbox="pipeline-lightbox" aria-label="Expand the Luciole compression pipeline">
  <img class="pipeline-figure" src="/assets/images/luciole-distillation/pipeline.svg" alt="Luciole compression pipeline from a 23.22B base model to an 8.36B distilled student" />
</button>

<dialog id="pipeline-lightbox" class="image-lightbox" aria-label="Expanded Luciole compression pipeline">
  <button class="image-lightbox-close" type="button" aria-label="Close expanded image">&times;</button>
  <img class="image-lightbox-image" src="/assets/images/luciole-distillation/pipeline.svg" alt="Luciole compression pipeline from a 23.22B base model to an 8.36B distilled student" />
</dialog>

<!-- *the model ladder. continued pretraining makes the teacher better at maths first; the two prune-and-distil stages then do the compression.* -->

## forward KL distillation

after pruning, the student has inherited weights but a different shape. it cannot just continue where the teacher left off. hence distillation.

at each token, the teacher gives a distribution over the vocabulary, $p_T$, and the student produces its own, $p_S$. i trained the student with forward KL:

$$
\mathcal{L}_{\mathrm{KD}} = D_{\mathrm{KL}}(p_T \parallel p_S)
= \sum_{v \in V} p_T(v) \log \frac{p_T(v)}{p_S(v)}.
$$

the useful intuition is that the target is not only the one token that appeared in the dataset. the student also gets the teacher's relative confidence in all the alternatives. because the teacher is fixed, minimising this is the same as cross-entropy against the teacher distribution, up to a constant.

i used pure logit distillation here: the hard-label next-token cross-entropy term was off. that made the training signal very explicit: reproduce the specialised teacher's token distribution. feature matching was not very attractive because pruning changes both depth and hidden width, so there is no neat one-to-one mapping between the student and teacher internals.

each full stage ran for 112b tokens. that is a lot of training, but it was still much cheaper than training another model from scratch, and it let the smaller models inherit the maths capability that had already been built into the teacher.

## selecting a pruning candidate

my initial instinct was simple: prune a few candidate architectures, measure which one is least broken immediately after the cut, and distil that one. it sounds reasonable. it was also wrong for this setup.

in the first stage, candidate 5 had the best mathematical perplexity straight after pruning: 1,276. candidate 4 started much worse, at 2,660. after only 100 distillation iterations, candidate 4 was the clear winner on both held-out maths and general text: 3.79 maths perplexity, while candidate 5 was at 9.47.

<div id="selection-probe-chart" class="plotly-figure plotly-figure-tall" aria-label="Short distillation probe showing candidate 4 recovering better than the least-damaged candidate"></div>

*candidate 5 looks best at iteration 0, but candidate 4 learns the recovery objective much better. step-0 perplexity was measuring pruning damage, not recoverability.*

so i changed the selection protocol. instead of ranking candidates immediately after pruning, i ran short distillation probes on a diverse small set and compared held-out perplexity after they had actually begun to learn. at the second stage this also made the search much more practical: dropping an unhelpful in-search scoring pass reduced the pruning job from roughly 15 hours to roughly 2.

this was probably the most useful data-story result from the project. a pruned model can look terrible and recover well; another can look relatively fine and learn badly. the architecture-selection metric needs to see the recovery process, not just the initial injury.

## what the 8b model kept

the final Luciole-8B reached 49.79 on the five-task mathematics average, compared with 50.17 for the corrected 23B teacher. it also beat the independently trained Luciole-8B base model by 18.06 points (31.73 to 49.79). on French generative MathALEA it reached 73.38, above both the teacher (69.75) and the from-scratch 8B model (52.67).

| model | parameters | maths average | general average |
| --- | ---: | ---: | ---: |
| corrected Luciole teacher | 23.22B | 50.17 | 66.53 |
| Luciole-14B-KD | 13.98B | 50.78 | 63.87 |
| Luciole-8B-KD | 8.36B | 49.79 | 61.71 |
| Luciole-8B-Base | 8.00B | 31.73 | 63.84 |

<div id="retention-chart" class="plotly-figure" aria-label="Math and general capability retained by the compressed Luciole models relative to the corrected teacher"></div>

*the final 8.36b model retains 99.2% of the teacher's mathematical score and 92.8% of its general score. the from-scratch 8b model makes the value of specialisation followed by distillation pretty visible.*

the trade-off is real. on held-out mathematical text, the 8b model finished slightly better than the teacher in perplexity (1.402 vs 1.436). on FineWeb-Edu, it finished worse (7.407 vs 5.607). the general benchmark average follows the same pattern: 61.71 for the 8b model against 66.53 for the teacher.

i would not call that catastrophic forgetting: the model is still broadly functional on general multiple-choice benchmarks. but it is a clear reminder that a maths-heavy distillation mixture and a smaller model both push capacity away from general text.

## future work?

the obvious next experiment is not "more distillation". it is a matched-budget final phase with more general text in the mixture, possibly with a small hard-label cross-entropy term, to see if some general performance can be recovered without giving the maths gains back.

i would also search more broadly over pruning ratios and candidate shapes. this project gives a good reason to use short KD probes for selection, but it only fully trained one winner at each stage. there is a lot more room to understand what makes a pruned architecture recover well.
