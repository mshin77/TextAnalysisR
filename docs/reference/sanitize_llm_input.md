# Sanitize LLM Input

Sanitizes user input before inclusion in LLM prompts to mitigate prompt
injection attacks. Filters common injection patterns such as instruction
overrides, system prompt markers, and role-switching attempts. Distinct
from
[`sanitize_text_input()`](https://mshin77.github.io/TextAnalysisR/reference/sanitize_text_input.md)
which targets XSS.

## Usage

``` r
sanitize_llm_input(text, max_length = 2000)
```

## Arguments

- text:

  Character string of user input destined for an LLM prompt

- max_length:

  Maximum allowed character length (default: 2000)

## Value

Sanitized character string

## NIST Compliance

Implements NIST SI-10 (Information Input Validation) for AI/LLM
contexts.
