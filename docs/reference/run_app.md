# Launch the TextAnalysisR app

Launch the TextAnalysisR Shiny application.

## Usage

``` r
run_app(launch.browser = interactive())
```

## Arguments

- launch.browser:

  Logical. Whether to open the app in a browser. Defaults to
  [`interactive()`](https://rdrr.io/r/base/interactive.html), which is
  FALSE in non-interactive sessions (e.g., Docker containers, servers).

## Value

No return value, called for side effects (launching Shiny app)

## Examples

``` r
if (interactive()) {
  library(TextAnalysisR)
  run_app()
}
```
