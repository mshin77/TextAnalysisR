# Detect Corpus Language With an LLM

Identifies the dominant language of a text sample by asking an LLM
(OpenAI or Gemini) to pick one name from a candidate list. More accurate
than
[`detect_language()`](https://mshin77.github.io/TextAnalysisR/reference/detect_language.md)
on short texts, mixed-language corpora, or languages with sparse
snowball stopword coverage, at the cost of an API call.

## Usage

``` r
detect_language_llm(
  texts,
  languages,
  provider = c("auto", "openai", "gemini"),
  model = NULL,
  api_key = NULL,
  sample_n = 200,
  seed = 123,
  verbose = TRUE
)
```

## Arguments

- texts:

  Character vector of documents.

- languages:

  Named character vector of candidate languages (name = display name,
  value = language code), e.g. `c(English = "en", French = "fr")`.

- provider:

  One of `"auto"`, `"openai"`, `"gemini"`. `"auto"` picks whichever of
  `OPENAI_API_KEY` / `GEMINI_API_KEY` is set, or the key implied by
  `api_key`'s prefix.

- model:

  Model name (default depends on provider).

- api_key:

  API key; falls back to the provider's environment variable.

- sample_n:

  Maximum documents to sample for the prompt (default 200).

- seed:

  Seed for sampling.

- verbose:

  Logical, print status messages (default TRUE).

## Value

A one-row tibble with `language` (code) and `provider`, or `NULL` when
no provider/key is available or the response doesn't match a candidate.

## See also

[`detect_language()`](https://mshin77.github.io/TextAnalysisR/reference/detect_language.md)
for the local, no-API alternative;
[`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md)
for the direct provider call.

## Examples

``` r
if (interactive()) {
detect_language_llm(
  c("Bonjour, comment allez-vous?", "Je suis ravi de vous rencontrer."),
  languages = c(English = "en", French = "fr", Spanish = "es"),
  provider = "openai",
  api_key = Sys.getenv("OPENAI_API_KEY")
)
}
```
