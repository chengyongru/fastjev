"""Direct categorical decision readout from native next-token logits."""

from __future__ import annotations

import inspect
import time

from .core import LETTERS, digest, direct_messages, softmax, synchronize

PROMPT_VERSION = "direct-options-v1"


def _slot_ids(tokenizer, count: int) -> list[int]:
    result = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        result.append(encoded[0])
    if len(result) != len(set(result)):
        raise ValueError("Answer-slot tokens collide")
    return result


def _forward(model, inputs):
    parameters = inspect.signature(model.forward).parameters
    kwargs = dict(inputs, use_cache=False, return_dict=True)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    return model(**kwargs).logits[:, -1, :]


def encode_prompt(tokenizer, row: dict, max_tokens: int) -> tuple[list[int], list[int], str]:
    """Encode one decision and verify its single-token answer slots."""
    prompt = tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; no truncation allowed")
    slots = _slot_ids(tokenizer, len(row["options"]))
    for letter, token in zip(LETTERS, slots):
        if tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + [token]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
    return ids, slots, digest(prompt)


def score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    import torch

    started = time.perf_counter()
    ids, slots, prompt_hash = encode_prompt(tokenizer, row, max_tokens)
    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
    }
    synchronize(device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        vocabulary = _forward(model, inputs)[0].float()
    synchronize(device)
    selected = vocabulary[slots].cpu().tolist()
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "input_tokens": len(ids),
        "forward_seconds": time.perf_counter() - forward_start,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }


def score_batch(model, tokenizer, rows: list[dict], metadata: dict, max_tokens: int = 4096,
                *, batch_size: int = 8, sort_by_length: bool = False) -> list[dict]:
    """Score left-padded prompts in bounded batches and restore the supplied order.

    Length sorting buffers at most eight batches of encoded prompts. Per-row timing
    reports the shared forward duration and that duration plus window encoding time;
    these values are not additive across rows. Measure call wall time for throughput.
    """
    import torch

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(sort_by_length) is not bool:
        raise ValueError("sort_by_length must be a boolean")
    if not rows:
        return []
    device = next(model.parameters()).device
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    # Masked padding can use any valid vocabulary entry without changing the tokenizer.
    if pad_id is None:
        pad_id = 0
    results = [None] * len(rows)
    window_size = batch_size * (8 if sort_by_length else 1)
    for start in range(0, len(rows), window_size):
        encode_start = time.perf_counter()
        encoded = [
            (index, *encode_prompt(tokenizer, rows[index], max_tokens))
            for index in range(start, min(start + window_size, len(rows)))
        ]
        encode_seconds = time.perf_counter() - encode_start
        if sort_by_length:
            encoded.sort(key=lambda item: len(item[1]))
        for offset in range(0, len(encoded), batch_size):
            batch = encoded[offset:offset + batch_size]
            width = max(len(item[1]) for item in batch)
            ids = [[pad_id] * (width - len(item[1])) + item[1] for item in batch]
            masks = [[0] * (width - len(item[1])) + [1] * len(item[1]) for item in batch]
            inputs = {
                "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
                "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
            }
            # Real tokens retain their unpadded positions, including for absolute embeddings.
            if any(len(item[1]) != width for item in batch):
                positions = inputs["attention_mask"].cumsum(-1) - 1
                inputs["position_ids"] = positions.masked_fill(inputs["attention_mask"] == 0, 0)
            synchronize(device)
            forward_start = time.perf_counter()
            with torch.inference_mode():
                vocabulary = _forward(model, inputs).float()
            synchronize(device)
            selected = [vocabulary[i, item[2]].cpu().tolist() for i, item in enumerate(batch)]
            forward_seconds = time.perf_counter() - forward_start
            for (index, tokens, slots, prompt_hash), logits in zip(batch, selected):
                row = rows[index]
                results[index] = {
                    "id": row["id"],
                    "option_ids": [option["id"] for option in row["options"]],
                    "probabilities": softmax(logits),
                    "option_logits": logits,
                    "input_tokens": len(tokens),
                    "forward_seconds": forward_seconds,
                    "total_seconds": encode_seconds + forward_seconds,
                    "prompt_sha256": prompt_hash,
                    "prompt_version": PROMPT_VERSION,
                    "model": metadata,
                    "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
                    "probability_status": "conditional option score; uncalibrated as decision confidence",
                }
    return results
