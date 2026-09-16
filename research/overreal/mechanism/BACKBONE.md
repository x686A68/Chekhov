# 6.3 "Diffusion backbone": does the backbone treat a mention as a request?

Design 2026-09-16. Section 6.3 asks whether the target's appearance is
caused by the encoder or by the backbone; after the encoder probe was
dropped from the paper (TEXT_ENCODER.md), this paragraph carries the section.

## Claim to test

The backbone reads the target token as if the prompt asked for it: the
target gets the same attention whether or not the prompt calls for it, that
attention is what puts the target into the picture, and the words that keep
the target out (no, believes, like, behind) are barely read.

## Data

- Prompts: the 545 S/P pairs in `pairs.jsonl` (S = original benchmark
  prompt, P = hand-written plain-mention control with the same target word).
- Seeds 0 and 1, steps and cfg taken from the benchmark manifests, so the S
  images are the benchmark images (FLUX reproduces them pixel-exactly; SD3.5
  within numerical noise) and the Opus labels apply.
- Models: FLUX.1-dev (57 blocks, T5 512 tokens), SD3.5-Large (38 blocks,
  77 CLIP + 256 T5 tokens), Qwen-Image (60 blocks, Qwen2.5-VL tokens).

## Recording (attn_common.py, attn_flux.py, attn_sd35.py, attn_qwen.py)

The generation path is untouched (SDPA). From the same q and k, at every
block and step, the attention probability from every image token to every
text token is recomputed on the side, averaged over heads, and reduced to
  mass [blocks, steps, n_txt]   mean over image tokens
  map  [5 step bins, 4096]      attention to the target tokens per image token
Target tokens: the target span of pairs.jsonl. Cue tokens: the words that
differ between S and P (word-level diff), i.e. the phrase the rewrite
replaced. Per-token quantities use the share of image-to-text attention on
the real tokens (pad, BOS, EOS and template tokens excluded), so images with
different total text attention are comparable.

## Readings (analyze_attn.py)

1. S vs P, paired by item and seed: log(share_target_S / share_target_P).
   Support: mean log-ratio near 0 (ratio above 0.9) in every family.
2. Within S: AUC of the target's share (and early-step share, and map peak)
   for over-realized (disruptive + silent) versus withheld images, labels
   from `final__claude-opus-5-api__sample.jsonl`. Support: AUC clearly above
   0.5. If near 0.5, reading 1 says nothing.
3. Cue tokens: per-token share against the other real tokens; and AUC of the
   cue share for withheld versus over-realized images. Support for "barely
   read": cue/other ratio at or below 1 and AUC near 0.5. If withheld images
   have clearly higher cue attention, the backbone does read the cue
   sometimes and that is how withholding happens: report as found.

## Outputs

`attn/<model>/<item>__<side>_s<seed>.{npz,png}`, `results/attn_summary_<model>.json`,
`results/attn_profile_<model>.json` (block x step share of the target, S and P).
The target-token heatmap of the fierce-coach tiger (figurative/prompt_0001)
is the candidate figure for the paragraph.

## Limits to state

Attention mass is the route, not the cause; DeLeaker (Ventura et al.) is the
precedent for reading and intervening on these maps. FLUX and SD3.5 also
receive a CLIP pooled vector through modulation, a channel this analysis
cannot see.

## Cost

FLUX: about 40 s per 50-step image with recording; 2,180 images per model.
SD3.5-L similar; Qwen-Image slower. Runs chained per GPU by
`run_attn_chain.sh` (FLUX, then SD3.5-L, then Qwen-Image).

## Results (runs finished 2026-09-17; three models, 2,180 records each)

Reading 1, target share ratio original/control (exp of mean paired log-ratio),
EC / MS / Fig / Per: FLUX .79 .74 .83 .98; SD3.5-L .85 .85 .92 .99;
Qwen-Image .80 .90 .92 .97. Every CI excludes 0 except perspectival.

Reading 2, AUC over-realized (DO+SO) vs withheld, early-step map peak:
FLUX .65 .69 .70 .56; SD3.5-L .92 .98 .87 .67; Qwen-Image .54 .87 .62 .48.
Mean share AUC is lower for SD3.5-L (.60 .84 .72 .51) and similar for FLUX.
Perspectival at chance everywhere.

Reading 3, replaced-words per-token share / other words: FLUX .71 .73 .77 .83;
SD3.5-L .46 .54 .63 .81; Qwen .73 .54 .54 .66. AUC of that share for
withheld vs over-realized: .33 to .57 (no effect).

Expanded prompts (FLUX, 25 labeled items per family with the target kept):
target per-token share relative to other words rises from .85 to 1.37 (Qwen
rewriter) and .97 to 1.43 (Ideogram); AUC of the relative share for
appearance .75 / .77 pooled.

Paper: section 6.3 (four paragraphs, Table tab:attn, Figure fig:attn with the
cucumber pair), Appendix app:encoder (probe restored) and app:attn (detail
tables). Example maps: results/examples/ (five appendix items on SD3.5-L,
cucumber pair in results/examples/paper/).
