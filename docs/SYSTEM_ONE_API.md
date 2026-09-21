# System One-compatible API

fastjev can expose a wire-compatible subset of TypeSafe's documented System One HTTP API. This adapter accepts the same top-level `state`, `model`, and `questions` fields at `POST /v1/systemone`, and returns `model`, `answers`, and `usage`. It is an independent compatibility layer: it does not run Jev, and fastjev scores are not calibrated as Jev probabilities.

The implemented contract follows the public [TypeSafe API reference](https://docs.typesafe.ai/api):

- `noul` maps to two fastjev options and returns the probability assigned to `true`.
- `choice` returns the highest-probability option and the complete option distribution.
- `score` treats the ordered criteria as levels `0..N-1` and returns their probability-weighted value.
- Multiple questions share one request state and are scored independently with fastjev's direct option-logit path. The experimental shared-prefix scorer is not exposed by this service because the repository does not claim that its decisions are semantically equivalent to direct scoring.
- `GET /v1/models` returns the single configured fastjev model.

## Install and run

Install the API extra in the same isolated environment as fastjev:

```bash
pip install -e '.[api,torch]'
```

Keep the default loopback binding for a local unauthenticated service:

```bash
CUDA_VISIBLE_DEVICES=0 fastjev-serve \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --served-model fastjev-qwen3.5-4b \
  --served-model-description 'fastjev direct option-logit baseline on Qwen3.5-4B BF16' \
  --served-model-release-date 2026-09-18
```

For a non-loopback bind, set a bearer token through the named environment variable. The server rejects an unauthenticated non-loopback bind unless `--allow-unauthenticated` is explicit.

```bash
export FASTJEV_API_KEY='replace-with-a-secret'
CUDA_VISIBLE_DEVICES=0 fastjev-serve \
  --host 0.0.0.0 \
  --model /path/to/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --served-model fastjev-qwen3.5-4b \
  --served-model-description 'fastjev direct option-logit baseline on Qwen3.5-4B BF16' \
  --served-model-release-date 2026-09-18
```

The process intentionally runs one Uvicorn worker. Additional workers would each load another model copy. Requests are serialized around the resident model so concurrent HTTP handlers do not race one GPU model.

## Request

```json
{
  "model": "fastjev-qwen3.5-4b",
  "state": "Help! My payouts have been failing for three days.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?"
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, and refunds",
        "technical": "Bugs, outages, and integrations"
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the problem?",
      "criteria": ["Minor", "Degraded", "Blocking"]
    }
  }
}
```

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @request.json
```

When authentication is configured, also send `Authorization: Bearer $FASTJEV_API_KEY`.

## Compatibility boundaries

The adapter preserves the public wire shape, not Jev's model behavior:

- The request `model` must equal the configured fastjev model ID. Jev aliases such as `jev-latest` are rejected instead of being impersonated.
- fastjev currently supports 2-16 options for `choice`; TypeSafe documents up to 255.
- `score` supports the documented 2-10 levels. `noul` and `choice` criteria may use strings, JSON objects, arrays, or `null`; structured content is rendered as JSON for the fastjev prompt.
- TypeSafe's SDK permits omitted or `null` instructions. The adapter supplies a type-specific generic question in that case; explicit instructions remain preferable because they define the intended decision boundary.
- fastjev's default input limit is 4,096 tokens per converted question, with no truncation. TypeSafe documents a different context budget.
- Choice and Score require a `confidence` field in the documented response. Because TypeSafe does not publish its exact statistic, the adapter returns `1 - normalized entropy` and identifies it as `one-minus-normalized-entropy` in the top-level `fastjev` extension. It is not numerically comparable to TypeSafe confidence.
- `usage.input_tokens` is the sum of fastjev's per-question prompt lengths. `usage.output_tokens` is zero because fastjev reads option logits without generating answer text.
- The returned `fastjev.probability_status` records that option probabilities are conditional and uncalibrated. Validate thresholds on the deployment workload before automating consequential actions.

FastAPI also exposes generated OpenAPI documentation at `/docs` and `/openapi.json`. These convenience routes are fastjev extensions, not TypeSafe endpoints.
