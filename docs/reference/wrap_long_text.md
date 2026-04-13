# Wrap Long Text with Line Breaks

Wraps long text by inserting line breaks at word boundaries. Handles
both spaced text and continuous text (like URLs).

## Usage

``` r
wrap_long_text(text, chars_per_line = 50, max_lines = 3)
```

## Arguments

- text:

  Character string to wrap.

- chars_per_line:

  Maximum characters per line (default: 50).

- max_lines:

  Maximum number of lines (default: 3).

## Value

Text with line breaks inserted.

## See also

Other text-utilities:
[`truncate_text_to_words()`](https://mshin77.github.io/TextAnalysisR/reference/truncate_text_to_words.md),
[`truncate_text_with_ellipsis()`](https://mshin77.github.io/TextAnalysisR/reference/truncate_text_with_ellipsis.md),
[`wrap_text_for_tooltip()`](https://mshin77.github.io/TextAnalysisR/reference/wrap_text_for_tooltip.md)
