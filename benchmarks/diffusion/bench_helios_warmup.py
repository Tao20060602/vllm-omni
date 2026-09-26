# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Measure Helios startup warmup and the first repeated video requests.

Run with a local model directory to keep downloads out of the timed process.
Set ``TORCH_LOGS=recompiles`` when diagnosing shape recompilation; those logs
are diagnostic and should not be used for final latency comparisons.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import vllm_omni
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Helios-Distilled model directory")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--warmup-height", type=int)
    parser.add_argument("--warmup-width", type=int)
    parser.add_argument("--warmup-num-frames", type=int)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--prompt",
        default="A serene lakeside sunrise with mist over the water",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to compare first and repeated requests")
    if args.height <= 0 or args.width <= 0 or args.num_frames <= 0:
        parser.error("height, width, and num-frames must be positive")
    for name in ("warmup_height", "warmup_width", "warmup_num_frames"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise ValueError(f"Use a pre-downloaded local model directory: {model_path}")

    warmup_shape = {
        name: value
        for name, value in (
            ("height", args.warmup_height),
            ("width", args.warmup_width),
            ("num_frames", args.warmup_num_frames),
        )
        if value is not None
    }
    omni_kwargs = {"model": str(model_path), "model_class_name": "HeliosPipeline", "enforce_eager": args.enforce_eager}
    if warmup_shape:
        omni_kwargs["additional_config"] = {"diffusion_warmup_shape": warmup_shape}

    print("[warmup-bench] startup begin", file=sys.stderr, flush=True)
    startup_start = time.perf_counter_ns()
    omni = Omni(**omni_kwargs)
    startup_ms = (time.perf_counter_ns() - startup_start) / 1_000_000
    print("[warmup-bench] startup end", file=sys.stderr, flush=True)

    prompt = {"prompt": args.prompt, "modalities": ["video"], "negative_prompt": ""}
    request_ms: list[float] = []
    try:
        for request_index in range(args.repeats):
            sampling = OmniDiffusionSamplingParams(
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed),
                extra_args={
                    "is_enable_stage2": True,
                    "pyramid_num_inference_steps_list": [2, 2, 2],
                    "is_amplify_first_chunk": True,
                },
            )
            print(f"[warmup-bench] request {request_index + 1} begin", file=sys.stderr, flush=True)
            request_start = time.perf_counter_ns()
            outputs = omni.generate(prompt, sampling, use_tqdm=False)
            elapsed_ms = (time.perf_counter_ns() - request_start) / 1_000_000
            print(f"[warmup-bench] request {request_index + 1} end", file=sys.stderr, flush=True)
            if not outputs or any(output.error for output in outputs):
                raise RuntimeError(f"Helios request failed: {[output.error for output in outputs]}")
            if not any(output.images for output in outputs):
                raise RuntimeError("Helios returned no video frames")
            request_ms.append(elapsed_ms)
            del outputs
    finally:
        omni.shutdown()

    result = {
        "model": str(model_path),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "warmup_shape": warmup_shape,
        "startup_ms": startup_ms,
        "startup_plus_first_request_ms": startup_ms + request_ms[0],
        "request_ms": request_ms,
        "torch_version": torch.__version__,
        "vllm_omni_source": str(Path(vllm_omni.__file__).resolve()),
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
