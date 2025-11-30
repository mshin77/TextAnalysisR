# Check Deployment Environment

Detects whether the app is running on a web server (shinyapps.io, Posit
Connect) versus locally via
[`run_app()`](https://mshin77.github.io/TextAnalysisR/reference/run_app.md).

## Usage

``` r
check_web_deployment()
```

## Value

Logical TRUE if running on web server, FALSE if local

## Examples

``` r
if (check_web_deployment()) {
  message("Running on web - some features disabled")
}
```
