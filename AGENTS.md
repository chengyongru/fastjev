# Repository instructions

- Format every commit message according to Conventional Commits: `<type>[optional scope]: <description>`. Keep the description short and imperative.
- Run commands from the repository root in an isolated environment installed with `pip install -e '.[test]'`.
- Validate changes with `pytest -q`, `(cd results/raw && sha256sum -c SHA256SUMS)`, and `python benchmarks/verify_published.py`.
- Benchmark outputs are create-only. Use a new output path and expose exactly one CUDA GPU per scorer process.
- Do not change headline claims or `results/phase1-summary.json` without committing the supporting row-level evidence, regenerating the relevant raw report, updating `results/raw/SHA256SUMS`, and updating the method/results text.
- Preserve exact model and source revisions. Use `benchmarks/fetch_sources.py` only for its listed redistributable inputs; do not commit model weights, caches, or third-party raw records.
- `webgpu-demo/` is static and has no build step. Preserve `_headers`, runtime version pins, browser-only inference, and the explicit probability limitations.
