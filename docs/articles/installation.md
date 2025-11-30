# Installation

## R Package

``` r
install.packages("remotes")
remotes::install_github("mshin77/TextAnalysisR")
library(TextAnalysisR)
run_app()
```

**Requirements:** R \>= 4.0, RStudio recommended

## Web App

Visit [textanalysisr.org](https://www.textanalysisr.org) - no
installation needed.

Note: Web version has limited features (no Python, no AI, no large
files).

## Optional Features

### Linguistic Analysis (spaCy)

For lemmatization, POS tagging, and named entity recognition:

``` r
install.packages("spacyr")
spacyr::spacy_install()
```

### Python Features

For PDF tables, embeddings, and AI-assisted analysis:

``` r
setup_python_env()
```

Requires Python 3.9+ and optionally [Ollama](https://ollama.ai) for
local AI.

## Troubleshooting

| Issue | Solution |
|----|----|
| Package install failed | `remotes::install_github("mshin77/TextAnalysisR", dependencies = TRUE)` |
| Browser doesn’t open | Navigate to URL shown in R console |
| Python errors | See [Python Environment](https://mshin77.github.io/TextAnalysisR/articles/python_environment.md) |
