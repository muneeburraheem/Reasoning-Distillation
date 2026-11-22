from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from batched_evals.generation import PROMPT_STYLES, SCORING_MODES


# Everything the harness writes (outputs, default cache) stays under this folder.
PKG_ROOT = Path(__file__).resolve().parent

SUPPORTED_BENCHMARKS = {
    "arc-c",
    "math500",
    "aime25",
    "gsm8k",
    "hellaswag",
    "mmlu",
    "gpqa",
    "openbookqa",
    "winogrande",
}


@dataclass
class ModelSpec:
    name: str
    checkpoint: str
    tokenizer: str | None = None
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    device_map: str = "auto"
    load_in_8bit: bool = False


@dataclass
class EvalConfig:
    models: list[ModelSpec]
    benchmarks: list[str]
    output_dir: str = "./eval_outputs"
    cache_dir: str | None = "./cache"
    split: str = "test"
    max_samples: int | None = None
    seed: int = 42
    cot: bool = False
    max_toks: int | None = None
    pass_at_k: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 128
    batch_size: int = 8
    prompt_style: str = "marker"
    mc_scoring: str = "generate"
    scoring_mode: str = "lenient"
    benchmark_options: dict[str, dict[str, Any]] = field(default_factory=dict)


def resolve_path(path_str: str | None, config_dir: Path) -> str | None:
    """Anchor relative paths at the YAML file's directory, not the cwd.

    Local checkpoints resolve only if the path exists (HF hub ids like
    'meta-llama/Llama-3.2-1B' look relative but must pass through untouched).
    """
    if path_str is None:
        return None
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return str(p)
    return str((config_dir / p).resolve())


def _resolve_checkpoint(ref: str, config_dir: Path) -> str:
    candidate = Path(resolve_path(ref, config_dir))
    if candidate.exists():
        return str(candidate)
    return ref  # assume HF hub id


def load_config(path: str | Path) -> EvalConfig:
    path = Path(path)
    config_dir = path.resolve().parent
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")

    raw_models = raw.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("Config must include a non-empty models list.")

    models = []
    for item in raw_models:
        if not isinstance(item, dict):
            raise ValueError("Each model entry must be a mapping.")
        checkpoint = item.get("checkpoint") or item.get("path") or item.get("model_id")
        if not checkpoint:
            raise ValueError("Each model needs checkpoint, path, or model_id.")
        tokenizer = item.get("tokenizer")
        models.append(
            ModelSpec(
                name=item.get("name") or Path(str(checkpoint)).name,
                checkpoint=_resolve_checkpoint(str(checkpoint), config_dir),
                tokenizer=_resolve_checkpoint(str(tokenizer), config_dir) if tokenizer else None,
                trust_remote_code=bool(item.get("trust_remote_code", True)),
                dtype=str(item.get("dtype", "bfloat16")),
                device_map=str(item.get("device_map", "auto")),
                load_in_8bit=bool(item.get("load_in_8bit", False)),
            )
        )

    benchmarks = [str(name).lower() for name in raw.get("benchmarks", [])]
    unknown = sorted(set(benchmarks) - SUPPORTED_BENCHMARKS)
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}")
    if not benchmarks:
        raise ValueError(f"At least one benchmark is required. Options: {sorted(SUPPORTED_BENCHMARKS)}")

    generation = raw.get("generation", {}) or {}
    if not isinstance(generation, dict):
        raise ValueError("generation must be a mapping.")

    prompt_style = str(generation.get("prompt_style", raw.get("prompt_style", "marker"))).lower()
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(
            f"generation.prompt_style must be one of {sorted(PROMPT_STYLES)} (got {prompt_style!r})."
        )

    mc_scoring = str(raw.get("mc_scoring", "generate")).lower()
    if mc_scoring not in {"generate", "loglikelihood"}:
        raise ValueError(
            f"mc_scoring must be 'generate' or 'loglikelihood' (got {mc_scoring!r})."
        )

    # Applies to the generative benchmarks only; log-likelihood scoring never
    # parses output. 'lenient' is the validated headline policy, 'strict' the
    # lower bound (see generation.extract_answer).
    scoring_mode = str(raw.get("scoring_mode", "lenient")).lower()
    if scoring_mode not in SCORING_MODES:
        raise ValueError(
            f"scoring_mode must be one of {sorted(SCORING_MODES)} (got {scoring_mode!r})."
        )

    cot = bool(generation.get("cot", raw.get("cot", False)))
    if prompt_style == "native" and cot:
        raise ValueError(
            "generation.cot is meaningless with prompt_style: native — the trained "
            "<think>…</think><answer>…</answer> format already produces reasoning "
            "and answer in one completion. Drop cot and raise max_new_tokens instead."
        )
    max_toks = generation.get("max_toks", raw.get("max_toks"))
    if cot and max_toks is None:
        raise ValueError("generation.max_toks is required when generation.cot is true.")
    if max_toks is not None and int(max_toks) <= 0:
        raise ValueError("generation.max_toks must be positive.")

    pass_at_k = int(generation.get("pass_at_k", raw.get("pass_at_k", 1)))
    if pass_at_k <= 0:
        raise ValueError("generation.pass_at_k must be positive.")

    batch_size = int(generation.get("batch_size", raw.get("batch_size", 8)))
    if batch_size <= 0:
        raise ValueError("generation.batch_size must be positive.")

    benchmark_options = raw.get("benchmark_options", {}) or {}
    if not isinstance(benchmark_options, dict):
        raise ValueError("benchmark_options must be a mapping.")

    # Defaults anchor at the package root so results/cache land inside
    # batched_evals/ no matter where the YAML lives or where you run from.
    output_dir = resolve_path(str(raw.get("output_dir", PKG_ROOT / "eval_outputs")), config_dir)
    if not Path(output_dir).resolve().is_relative_to(PKG_ROOT):
        raise ValueError(
            f"output_dir must stay inside {PKG_ROOT} (got {output_dir}); "
            "batched_evals is self-contained and never writes outside its folder."
        )

    return EvalConfig(
        models=models,
        benchmarks=benchmarks,
        output_dir=output_dir,
        cache_dir=(
            resolve_path(str(raw_cache), config_dir)
            if (raw_cache := raw.get("cache_dir", PKG_ROOT / "cache")) is not None
            else None
        ),
        split=str(raw.get("split", "test")),
        max_samples=raw.get("max_samples"),
        seed=int(raw.get("seed", 42)),
        cot=cot,
        max_toks=int(max_toks) if max_toks is not None else None,
        pass_at_k=pass_at_k,
        temperature=float(generation.get("temperature", 0.0)),
        top_p=float(generation.get("top_p", 1.0)),
        max_new_tokens=int(generation.get("max_new_tokens", 128)),
        batch_size=batch_size,
        prompt_style=prompt_style,
        mc_scoring=mc_scoring,
        scoring_mode=scoring_mode,
        benchmark_options=benchmark_options,
    )
