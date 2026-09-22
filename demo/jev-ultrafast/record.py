"""Record an official Jev Ultrafast case with FastJev decisions."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from browser_harness.helpers import drain_events

import jev_ultrafast
from jev_ultrafast import Agent
from jev_ultrafast import model as jev_model

TYPE_SAFE_SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("output", type=Path, help="New directory for frames and trace evidence")
    result.add_argument("--case", choices=("travel", "flights"), default="travel")
    result.add_argument(
        "--end-to-end",
        action="store_true",
        help="Give all published travel steps to one upstream Agent goal instead of advancing them one at a time",
    )
    result.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/systemone")
    result.add_argument("--model", default="fastjev-qwen3.5-4b")
    result.add_argument("--api-key", default="local-demo")
    result.add_argument("--text-endpoint", default="http://127.0.0.1:8001/v1")
    result.add_argument("--text-model", required=True)
    result.add_argument("--model-source", default="Qwen/Qwen3.5-4B")
    result.add_argument("--model-revision", required=True)
    result.add_argument("--fastjev-revision", required=True)
    result.add_argument("--jev-ultrafast-revision", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frames = output / "frames"
    screencast = output / "screencast"
    frames.mkdir()
    screencast.mkdir()

    upstream_root = Path(jev_ultrafast.__file__).resolve().parent.parent
    sys.path.insert(0, str(upstream_root))
    if args.case == "flights":
        from examples.flights import GOALS, URL, verify  # noqa: PLC0415
    else:
        # Keep the exact ordered task published in jev-ultrafast's
        # docs/measurement.json rather than substituting a custom prompt.
        GOALS = (
            'Enter "Lisbon" into the Destination field.',
            "Click Find stays to search for Lisbon.",
            "Set Stay category to Design.",
            "Enable Free cancellation.",
            "Open Casa Flora.",
        )
        URL = "http://127.0.0.1:8766/fixture.html?scenario=travel"

        def verify(page: dict) -> dict:
            expected = "Your filters: Design · Free cancellation enabled · Destination Lisbon"
            checks = {
                "property": page["url"].endswith("#casa-flora") and "Casa Flora" in page["text"],
                "category": expected in page["text"],
                "free_cancellation": "Free cancellation included" in page["text"],
                "destination": "Destination Lisbon" in page["text"],
            }
            return {"passed": all(checks.values()), "checks": checks, "final_url": page["url"]}

    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (upstream_root / "jev_ultrafast").iterdir()
        if path.suffix in {".py", ".js"}
    }
    provider_responses: list[dict] = []
    text_responses: list[dict] = []
    original_post_json = jev_model.post_json

    def post_json(url: str, key: str, body: dict) -> dict:
        if url == TYPE_SAFE_SYSTEM_ONE_URL:
            response = original_post_json(args.endpoint, args.api_key, body)
            provider_responses.append(response)
            return response
        if url == args.text_endpoint.rstrip("/") + "/chat/completions":
            local_body = {name: value for name, value in body.items() if name != "reasoning"}
            response = original_post_json(url, key, local_body)
            text_responses.append(response)
            return response
        return original_post_json(url, key, body)

    # Upstream fixes the TypeSafe URL in choose(). Replacing only its transport
    # preserves the official policy, task, page, and independent verifier.
    jev_model.post_json = post_json
    os.environ.update(
        TYPESAFE_API_KEY=args.api_key,
        TYPESAFE_MODEL=args.model,
        TEXT_MODEL_API_KEY="local-demo",
        TEXT_MODEL_BASE_URL=args.text_endpoint,
        TEXT_MODEL=args.text_model,
        TEXT_MODEL_REASONING="none",
    )

    agent = Agent(URL, GOALS)
    official_steps = list(GOALS) if args.case == "travel" and not args.end_to_end else []
    if official_steps:
        agent.state.update(plan=official_steps, plan_index=0, goal=official_steps[0])
    (frames / "000000.jpg").write_bytes(
        base64.b64decode(agent.browser.call("Page.captureScreenshot", format="jpeg", quality=85)["data"])
    )
    stop = threading.Event()
    recording_errors: list[str] = []
    epoch = time.time()

    def capture() -> None:
        try:
            while not stop.is_set():
                for event in drain_events():
                    if event["method"] != "Page.screencastFrame" or event.get("session_id") != agent.browser.session:
                        continue
                    params = event["params"]
                    timestamp = max(0, round((params["metadata"]["timestamp"] - epoch) * 1000))
                    (screencast / f"{timestamp:06d}.jpg").write_bytes(base64.b64decode(params["data"]))
                    agent.browser.call("Page.screencastFrameAck", sessionId=params["sessionId"])
                stop.wait(0.015)
        except Exception as error:  # evidence is retained and the run fails below
            recording_errors.append(str(error))

    agent.browser.call(
        "Page.startScreencast",
        format="jpeg",
        quality=80,
        maxWidth=1120,
        maxHeight=780,
        everyNthFrame=2,
    )
    worker = threading.Thread(target=capture, daemon=True)
    worker.start()
    epoch = time.time()
    state: dict = {}
    run_error: str | None = None
    try:
        if official_steps:
            for index, goal in enumerate(official_steps):
                agent.state.update(goal=goal, status="ready", plan_index=index)
                for state in agent.run():
                    action = state["history"][-1]["action"] if state["history"] else state["status"]
                    print(state["elapsed_ms"], f"step {index + 1}/{len(official_steps)}", state["status"], action, flush=True)
                if agent.state["status"] != "done":
                    break
                agent.state["plan_index"] = index + 1
        else:
            for state in agent.run():
                action = state["history"][-1]["action"] if state["history"] else state["status"]
                print(state["elapsed_ms"], state["status"], action, flush=True)
    except Exception as error:
        run_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        time.sleep(0.08)
        stop.set()
        worker.join(timeout=3)
        agent.browser.call("Page.stopScreencast")
        state = agent.snapshot()
        final_page = agent.browser.observe(screenshot=False)
        verification = verify(final_page)
        state.update(
            final_page=final_page,
            verification=verification,
            source_hashes=source_hashes,
            recording_errors=recording_errors,
            run_error=run_error,
            fastjev_provider_responses=provider_responses,
            text_provider_responses=text_responses,
            source_revisions={
                "fastjev": args.fastjev_revision,
                "jev_ultrafast": args.jev_ultrafast_revision,
                "model": args.model_revision,
            },
        )
        (output / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        (output / "session.json").write_text(
            json.dumps({"target": agent.browser.target, "session": agent.browser.session}), encoding="utf-8"
        )
        agent.close()

    provider_checks = {
        "status_done": state["status"] == "done",
        "response_count_matches_decisions": len(provider_responses) == len(state["decisions"]),
        "model_identity": all(response.get("model") == args.model for response in provider_responses),
        "fastjev_extension": all(isinstance(response.get("fastjev"), dict) for response in provider_responses),
        "zero_output_tokens": all(
            response.get("usage", {}).get("output_tokens") == 0 for response in provider_responses
        ),
    }
    summary = {
        "elapsed_ms": state["elapsed_ms"],
        "actions": len(state["history"]),
        "decisions": len(state["decisions"]),
        "goal": GOALS,
        "official_case": args.case,
        "model": args.model,
        "model_source": args.model_source,
        "model_revision": args.model_revision,
        "text_model": args.text_model,
        "verification": {
            **state["verification"],
            **provider_checks,
            "fastjev_responses": len(provider_responses),
            "type_safe_api_calls": 0,
            "fastjev_output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in provider_responses),
        },
        "source_revisions": state["source_revisions"],
        "recording_errors": recording_errors,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not state["verification"]["passed"] or not all(provider_checks.values()) or recording_errors:
        raise SystemExit("Recorded run failed independent verification")


if __name__ == "__main__":
    main()
