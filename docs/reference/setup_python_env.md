# Setup Python Environment

Intelligently sets up Python virtual environment with required packages.
Detects existing Python installations and guides users if Python is
missing.

## Usage

``` r
setup_python_env(envname = "textanalysisr-env", force = FALSE)
```

## Arguments

- envname:

  Character string name for the virtual environment (default:
  "textanalysisr-env")

- force:

  Logical, whether to recreate environment if it exists (default: FALSE)

## Value

Invisible TRUE if successful, stops with error message if failed

## Details

This function:

- Automatically detects if Python is already installed

- Offers to install Miniconda if no Python found

- Creates an isolated virtual environment (does NOT modify system
  Python)

- Installs minimal core packages:

  - spacy (NLP processing)

  - pdfplumber (PDF table extraction)

- Dependencies installed automatically by pip

- Avoids heavy packages (no torch, transformers)

The virtual environment approach means:

- No conflicts with other Python projects

- Easy to remove (just delete the environment)

- System Python remains untouched

- Much smaller download (~100MB vs 5GB+)

After setup, restart R session to activate enhanced features.

## Examples

``` r
if (FALSE) { # \dontrun{
# First time setup (auto-detects Python)
setup_python_env()

# Recreate environment
setup_python_env(force = TRUE)
} # }
```
