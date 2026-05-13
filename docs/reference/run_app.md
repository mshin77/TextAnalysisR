# Launch the TextAnalysisR app

Launch the TextAnalysisR Shiny application.

## Usage

``` r
run_app(launch.browser = interactive(), dev = FALSE)
```

## Arguments

- launch.browser:

  Logical. Whether to open the app in a browser. Defaults to
  [`interactive()`](https://rdrr.io/r/base/interactive.html), which is
  FALSE in non-interactive sessions (e.g., Docker containers, servers).

- dev:

  Logical. If `TRUE`, runs an accessibility palette audit (via
  [`a11yviz::a11y_check_palette()`](https://mshin77.github.io/a11yviz/reference/a11y_check_palette.html))
  on inline hex colors found in the Shiny source files and prints any
  pairs that fail WCAG 2.1 AA contrast against the app's light
  (`#ffffff`) and dark (`#0d1117`) backgrounds. Requires the `a11yviz`
  package.

## Value

No return value, called for side effects (launching Shiny app).

## Examples

``` r
if (interactive()) {
  library(TextAnalysisR)
  run_app()
}
```
