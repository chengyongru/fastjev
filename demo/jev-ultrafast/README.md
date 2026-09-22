# FastJev × Jev Ultrafast

[English](README.md) | [简体中文](README.zh-CN.md)

![FastJev driving the official Jev Ultrafast travel task](assets/fastjev-jev-ultrafast.gif)

This demo connects [Jev Ultrafast](https://github.com/browser-use/jev-ultrafast)
to FastJev's local System One-compatible API. Jev Ultrafast observes the page,
builds its indexed operation and target choices, executes the selected browser
action, and observes again. The recorder redirects only the System One endpoint
that upstream normally sends to TypeSafe.

The recording uses the official local travel fixture and the five ordered goals
published in Jev Ultrafast's `docs/measurement.json`:

1. Enter `Lisbon` into the Destination field.
2. Click **Find stays**.
3. Set **Stay category** to **Design**.
4. Enable **Free cancellation**.
5. Open **Casa Flora**.

One upstream `Agent` receives those strings as a single newline-delimited task.
No operation, target, field value, or completion result is hardcoded by the
recorder. A separate local text model supplies the destination after FastJev
selects `TYPE_TEXT`; every structured operation and target decision comes from
FastJev.

## Recorded result

The accepted run used one RTX 5090 and these pinned sources:

| Component | Exact source |
|---|---|
| FastJev runtime branch | `240230062901b92fba2ab79d5ecff751369ef7be` plus the System One adapter changes in this branch |
| Jev Ultrafast | `1231850a0bf1a0c0341fe408ef1668dbbfdfac46` |
| Decision model | `turboderp/Qwen3.8-27B-exl3` at `a35e75a73baee51da709329d19294245cbeeb5d8` |
| Text helper | `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

The agent completed all five browser actions and returned `DONE` on its sixth
FastJev request. The run took 11.776 seconds from the first decision request to
the verified result. Median decision-request latency was 1.456 seconds. The
first request included the cold EXL3 path; the separate text helper took 1.148
seconds to return `Lisbon`.

The independent verifier confirmed the final `#casa-flora` URL, destination,
Design category, enabled free-cancellation filter, and Casa Flora detail view.
All six FastJev responses reported zero output tokens, and the recorder made no
TypeSafe API calls.

These values describe one recorded integration run, not a latency benchmark.
They exclude model loading and browser startup and are not directly comparable
with Jev Ultrafast's published timings, which use a different decision model and
text-input path. FastJev probabilities are conditioned on the supplied choices
and are not calibrated as decision confidence.

Artifacts:

- [MP4 at original run speed](assets/fastjev-jev-ultrafast.mp4)
- [GIF preview](assets/fastjev-jev-ultrafast.gif)
- [Poster](assets/fastjev-jev-ultrafast-poster.png)
- [Verified action, decision, and usage trace](assets/verified-run.json)
- [Artifact checksums](assets/SHA256SUMS)

The 13.876-second presentation adds an 850 ms lead-in and a 1.25-second result
hold. The 11.776-second browser run remains at 1× speed.

## Reproduce

Install the FastJev runtime-capabilities branch and its EXL3 and API extras in an
isolated environment:

```bash
git checkout 240230062901b92fba2ab79d5ecff751369ef7be
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test,torch,api,exl3]'
```

Expose exactly one GPU and start the resident decision model:

```bash
CUDA_VISIBLE_DEVICES=0 fastjev-serve \
  --backend exl3 \
  --model turboderp/Qwen3.8-27B-exl3 \
  --revision a35e75a73baee51da709329d19294245cbeeb5d8 \
  --exl3-cache-size 8192 \
  --exl3-gpu-split 22.5 \
  --served-model fastjev-qwen3.8-27b-exl3 \
  --served-model-description 'FastJev direct option logits on Qwen3.8-27B EXL3' \
  --served-model-release-date 2026-09-22 \
  --host 127.0.0.1 \
  --port 8000 \
  --allow-unauthenticated
```

Check out Jev Ultrafast at the recorded revision, install its locked environment,
and start its loopback-only fixture server:

```bash
git clone https://github.com/browser-use/jev-ultrafast.git
cd jev-ultrafast
git checkout 1231850a0bf1a0c0341fe408ef1668dbbfdfac46
uv sync
uv run python -m jev_ultrafast.demo
```

Start an OpenAI-compatible text endpoint for the pinned Qwen3.5-4B snapshot on
port 8001. Start Chrome or Edge with a dedicated profile and a CDP endpoint,
then set `BU_CDP_WS` to the `webSocketDebuggerUrl` returned by
`http://127.0.0.1:9222/json/version`.

Run the recorder from the Jev Ultrafast environment. The output directory must
not already exist:

```bash
uv run python /path/to/fastjev/demo/jev-ultrafast/record.py /new/run-directory \
  --case travel \
  --end-to-end \
  --model fastjev-qwen3.8-27b-exl3 \
  --model-source turboderp/Qwen3.8-27B-exl3 \
  --model-revision a35e75a73baee51da709329d19294245cbeeb5d8 \
  --text-model /path/to/Qwen3.5-4B \
  --fastjev-revision 240230062901b92fba2ab79d5ecff751369ef7be \
  --jev-ultrafast-revision 1231850a0bf1a0c0341fe408ef1668dbbfdfac46
```

The recorder accepts a run only when the upstream agent reaches `done`, every
travel-task check passes, FastJev reports zero decision output tokens, no
TypeSafe call occurs, and screencast capture has no errors.

Render into new output paths with Pillow and FFmpeg available:

```bash
uv run --with pillow python /path/to/fastjev/demo/jev-ultrafast/render.py \
  /new/run-directory /new/render/fastjev-jev-ultrafast.mp4 \
  --gif /new/render/fastjev-jev-ultrafast.gif \
  --poster /new/render/fastjev-jev-ultrafast-poster.png
```

Verify the committed artifacts:

```bash
cd demo/jev-ultrafast/assets
sha256sum -c SHA256SUMS
```
