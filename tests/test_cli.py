import sys

import pytest

from fastjev.cli import main


@pytest.mark.parametrize("extra,message", [
    (["--backend", "mlx", "--mode", "reranker"], "reranker requires torch"),
    (["--backend", "exl3", "--mode", "serial"], "EXL3 supports direct mode only"),
    (["--mode", "direct", "--mlx-bits", "4"], "requires --backend mlx"),
    (["--mode", "direct", "--mlx-cache-limit-mib", "0"], "requires --backend mlx"),
    (["--mode", "direct", "--backend", "mlx", "--mlx-cache-limit-mib", "-1"], "must be nonnegative"),
    (["--mode", "direct", "--llama-cpp-filename", "model.gguf"], "requires --backend llama-cpp"),
])
def test_invalid_backend_combinations_fail_before_loading(tmp_path, monkeypatch, capsys, extra, message):
    monkeypatch.setattr(sys, "argv", ["fastjev-score", "--model", "unused", "--revision", "unused",
                                    "--input", "missing.jsonl", "--output", str(tmp_path / "out.jsonl"), *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.parametrize('limit', [None, 0, 512])
def test_cli_passes_cache_limit_to_loader(tmp_path, monkeypatch, limit):
    import json
    from types import SimpleNamespace
    from fastjev import _runtime

    fake_backend = SimpleNamespace(
        DEFAULT_CACHE_LIMIT_MIB=256,
        load_model=lambda source, revision, bits, *, cache_limit_mib:
            (None, None, {'limit': cache_limit_mib}),
        score=lambda model, tokenizer, row, metadata, max_tokens: metadata,
        SerialPrefixScorer=None, score_shared=None,
    )
    monkeypatch.setattr(_runtime, 'mlx', fake_backend)
    source, output = tmp_path / 'input.jsonl', tmp_path / 'output.jsonl'
    source.write_text(json.dumps({'id': 'test', 'state': 'Evidence', 'question': 'Supported?',
                                 'options': [{'id': 'yes', 'description': 'Yes'}, {'id': 'no', 'description': 'No'}]}) + '\n')
    args = ['fastjev-score', '--backend', 'mlx', '--mode', 'direct', '--model', 'unused',
            '--revision', 'unused', '--input', str(source), '--output', str(output)]
    if limit is not None:
        args += ['--mlx-cache-limit-mib', str(limit)]
    monkeypatch.setattr(sys, 'argv', args)
    main()
    assert json.loads(output.read_text())['limit'] == (256 if limit is None else limit)


def test_cli_passes_llama_cpp_options_to_loader(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from fastjev import _runtime

    observed = {}

    def load_model(source, revision, *, filename, max_input_tokens, n_batch, n_gpu_layers):
        observed.update(
            source=source,
            revision=revision,
            filename=filename,
            max_input_tokens=max_input_tokens,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
        )
        return None, None, {"backend": "fixture"}

    def score(_model, _tokenizer, row, metadata, max_tokens):
        return {
            "id": row["id"],
            "option_ids": [option["id"] for option in row["options"]],
            "probabilities": [0.5, 0.5],
            "metadata": metadata,
            "max_tokens": max_tokens,
        }

    fake_backend = SimpleNamespace(load_model=load_model, score=score)
    monkeypatch.setattr(_runtime, "llama_cpp", fake_backend)
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text(json.dumps({
        "id": "test",
        "state": "Evidence",
        "question": "Supported?",
        "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
    }) + "\n")
    monkeypatch.setattr(sys, "argv", [
        "fastjev-score", "--backend", "llama-cpp", "--mode", "direct",
        "--model", "fixture/repo", "--revision", "a" * 40, "--input", str(source),
        "--output", str(output), "--max-tokens", "123", "--llama-cpp-n-batch", "64",
        "--llama-cpp-n-gpu-layers", "7", "--llama-cpp-filename", "model-q4.gguf",
    ])

    main()

    assert observed == {
        "source": "fixture/repo",
        "revision": "a" * 40,
        "filename": "model-q4.gguf",
        "max_input_tokens": 123,
        "n_batch": 64,
        "n_gpu_layers": 7,
    }
    assert json.loads(output.read_text())["max_tokens"] == 123
