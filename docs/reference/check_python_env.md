# Check Python Environment Status

Checks if Python environment is available and properly configured.

## Usage

``` r
check_python_env(envname = "textanalysisr-env")
```

## Arguments

- envname:

  Character string name of the virtual environment (default:
  "textanalysisr-env")

## Value

List with status information:

- available: Logical, TRUE if environment exists

- active: Logical, TRUE if environment is currently active

- packages: List of installed package versions

## Examples

``` r
if (FALSE) { # \dontrun{
status <- check_python_env()
print(status)
} # }
```
