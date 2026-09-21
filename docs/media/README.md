# Apple Silicon CLI capture

![Completed recording of the native MLX CLI](openjev-mlx.png)

[Download the terminal recording](openjev-mlx.cast) and replay it with
`asciinema play openjev-mlx.cast`. This is an asciicast v2 recording of the
actual process output and timing, not a simulated terminal animation. The PNG
is a browser screenshot of the completed recording in asciinema-player 3.17.0.

Captured on 2026-09-17 with an Apple M5 Max, 128 GiB unified memory, Metal,
MLX 0.32.2, and pinned MLX-LM 0.32.0. The code under test was commit
`56e7ce2a38214137f476d2444e9193f72dcbb4ab`, before the configurable-cache
follow-up; this run used the unchanged 256 MiB default.

The fixture is `examples/decisions.jsonl`. The actual CLI output is retained in
[openjev-mlx-results.jsonl](openjev-mlx-results.jsonl), including model revision,
source hashes, probabilities, and timing metadata. The small displayed table
selects the largest recorded probability for each row; it is presentation of
the saved output, not additional inference. `1.0000` is rounded to four decimals.

This historical recording retains the former OpenJev name and command.
For the current fastjev checkout, run from the repository root after installing
the MLX extra:

```bash
fastjev-score --backend mlx --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl \
  --output results-mlx-demo.jsonl
```

The recorded output path is abbreviated to `<new demo output>` on screen;
choose a new path for each run. The checkpoint was already cached, network
access was disabled, and download progress bars were suppressed. The recorded
6.36-second wall time includes loading and hashing. It is a CLI demonstration,
not the throughput benchmark. Conditional option scores are uncalibrated.
