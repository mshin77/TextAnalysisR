# Truncate Text to Word Count

Truncates text to a maximum number of words and adds ellipsis if
truncated.

## Usage

``` r
truncate_text_to_words(text, max_words = 150)
```

## Arguments

- text:

  Character string to truncate.

- max_words:

  Maximum number of words (default: 150).

## Value

Truncated text with "..." appended if truncated.

## See also

Other text-utilities:
[`truncate_text_with_ellipsis()`](https://mshin77.github.io/TextAnalysisR/reference/truncate_text_with_ellipsis.md),
[`wrap_long_text()`](https://mshin77.github.io/TextAnalysisR/reference/wrap_long_text.md),
[`wrap_text_for_tooltip()`](https://mshin77.github.io/TextAnalysisR/reference/wrap_text_for_tooltip.md)
