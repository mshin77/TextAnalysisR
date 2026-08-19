# Apply a Codebook to Texts

Suggests codes for each unit of text from a supplied codebook using an
AI provider (OpenAI or Gemini). Documents are split into sentences,
paragraphs, or whole documents before coding, and each unit receives
zero or more codes. Output is a set of suggestions for human
confirmation, not final codes.

## Usage

``` r
apply_codes(
  texts,
  codebook,
  unit = c("paragraph", "sentence", "document"),
  max_codes = 3,
  provider = c("auto", "openai", "gemini"),
  model = NULL,
  temperature = 0,
  api_key = NULL,
  delay = 1,
  verbose = TRUE
)
```

## Arguments

- texts:

  Character vector of documents. Names become `doc_id`.

- codebook:

  Data frame with `code` and `definition` columns; optional `example`
  and `color`.

- unit:

  Unit of analysis: "sentence", "paragraph" (default, split on blank
  lines), or "document" (one unit per element of `texts`).

- max_codes:

  Maximum codes per unit (default 3). Units where no code fits return
  one row with `code = NA`.

- provider:

  AI provider: "auto" (default), "openai", or "gemini".

- model:

  Optional model id; provider default when NULL.

- temperature:

  Sampling temperature (default 0 for reproducibility).

- api_key:

  Optional API key; falls back to the provider env var.

- delay:

  Seconds to wait between provider calls (default 1).

- verbose:

  Logical; print per-unit progress (default TRUE).

## Value

A tibble with one row per unit-code pair: `doc_id`, `unit_id`, `start`,
`end` (character offsets of the unit within its document), `code`,
`confidence`, `rationale`. Returns `invisible(NULL)` when no provider
key is available.

## See also

[`code_retest()`](https://mshin77.github.io/TextAnalysisR/reference/code_retest.md)
for AI stability;
[`code_agreement()`](https://mshin77.github.io/TextAnalysisR/reference/code_agreement.md)
for inter-coder reliability;
[`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md)
for the direct provider call.
