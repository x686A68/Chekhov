"""Local T2I runner for the OverReal generation phases.

Phase 1:  --cond raw       (the item's own prompt)
Phase 2:  --cond qwen|dalle3|ideogram   (that expander's rewritten prompt;
          requires expanded_prompts.jsonl to already hold the texts)

One process drives one model on one GPU; run several in parallel, e.g.
  CUDA_VISIBLE_DEVICES=0 python run_local.py --model flux --cond raw
  CUDA_VISIBLE_DEVICES=1 python run_local.py --model qwen-image --cond raw

Qwen-Image is always fed the prompt text as-is: its official rewriter lives in
expand_qwen.py and is never invoked here, so "expansion off" holds by
construction for every model in this script.

Two seeds per prompt (common.SEEDS). Resumable: (item, seed) pairs already in
the manifest are skipped.
"""
import argparse
import datetime
import os
import time

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

from common import (CONDS, SEEDS, append_jsonl, done_keys, image_path,
                    load_prompts, manifest_path)

STEPS = {"flux": 50, "qwen-image": 50, "omnigen2": 50, "sd35m": 40, "sd35l": 40}
CFG = {"flux": 3.5, "qwen-image": 4.0, "omnigen2": 4.0, "sd35m": 4.5, "sd35l": 4.5}
SIZE = 1024


def load_pipe(model):
    import torch
    if model == "flux":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    elif model == "qwen-image":
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
    elif model == "sd35m":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
    elif model == "sd35l":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
    elif model == "omnigen2":
        # package lives in the cloned repo, not on PyPI (see ~/repos/OmniGen2)
        import sys
        sys.path.insert(0, os.path.expanduser("~/repos/OmniGen2"))
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.models.transformers import OmniGen2Transformer2DModel
        pipe = OmniGen2Pipeline.from_pretrained(
            "OmniGen2/OmniGen2", torch_dtype=torch.bfloat16, trust_remote_code=True)
        pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
            "OmniGen2/OmniGen2", subfolder="transformer", torch_dtype=torch.bfloat16)
    else:
        raise SystemExit(f"unknown model {model}")
    return pipe.to("cuda")


def call_pipe(pipe, model, prompt, steps, seed):
    import torch
    gen = torch.Generator("cuda").manual_seed(seed)
    kwargs = dict(prompt=prompt, num_inference_steps=steps, width=SIZE,
                  height=SIZE, generator=gen)
    if model == "omnigen2":                      # OmniGen2 names its cfg differently
        kwargs["text_guidance_scale"] = CFG[model]
    elif model == "qwen-image":                  # not guidance-distilled: true CFG,
        kwargs["true_cfg_scale"] = CFG[model]    # enabled by a negative prompt
        kwargs["negative_prompt"] = " "          # (official demo settings)
    else:
        kwargs["guidance_scale"] = CFG[model]
    return pipe(**kwargs).images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(STEPS))
    ap.add_argument("--cond", required=True, choices=CONDS)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N images")
    ap.add_argument("--steps", type=int, default=0, help="override step count (debug)")
    args = ap.parse_args()

    import torch

    prompts = load_prompts(args.cond)
    done = done_keys(args.model, args.cond)
    jobs = [(i, f, p, s) for i, f, p in prompts for s in SEEDS if (i, s) not in done]
    print(f"{args.model}/{args.cond}: {len(jobs)} images to go ({len(done)} done)")
    if not jobs:
        return

    pipe = load_pipe(args.model)
    steps = args.steps or STEPS[args.model]

    for n, (item_id, family, prompt, seed) in enumerate(jobs, 1):
        t0 = time.time()
        img = call_pipe(pipe, args.model, prompt, steps, seed)
        out = image_path(args.model, args.cond, item_id, seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        append_jsonl(manifest_path(args.model, args.cond), {
            "item_id": item_id, "family": family, "model": args.model,
            "cond": args.cond, "seed": seed, "steps": steps,
            "cfg": CFG[args.model], "size": SIZE, "prompt": prompt,
            "file": str(out.relative_to(out.parents[4])),
            "sec": round(time.time() - t0, 1),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        if n % 20 == 0 or n == len(jobs):
            print(f"[{n}/{len(jobs)}] {item_id} s{seed} {time.time()-t0:.1f}s", flush=True)
        if args.limit and n >= args.limit:
            print("limit reached, stopping")
            break


if __name__ == "__main__":
    main()
