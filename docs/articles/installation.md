# Installation

``` r

library(TextAnalysisR)
packageVersion("TextAnalysisR")
```

    ## [1] '0.1.4'

## R Package

``` r

install.packages("TextAnalysisR",
  repos = c("https://mshin77.r-universe.dev", "https://cloud.r-project.org"))
library(TextAnalysisR)
run_app()
```

**Requirements:** R \>= 4.0, RStudio recommended

## Web App

Visit [textanalysisr.org](https://www.textanalysisr.org) - no
installation needed.

Note: Web version has limited features (no Python, no local AI via
Ollama, no large files). Cloud AI providers (OpenAI, Gemini) are
available with an API key.

## Optional Features

### Linguistic Analysis (spaCy)

For lemmatization, POS tagging, and named entity recognition:

``` r

# Python spaCy required - see setup_python_env()
TextAnalysisR::setup_python_env()
```

### Python Features

For embeddings and neural sentiment analysis:

``` r

setup_python_env()
```

Requires Python 3.9+ and optionally [Ollama](https://ollama.com) for
local AI.

## Troubleshooting

| Issue | Solution |
|----|----|
| Package install failed | `remotes::install_github("mshin77/TextAnalysisR", dependencies = TRUE)` |
| Browser doesn’t open | Navigate to URL shown in R console |
| Python errors | See [Python Environment](https://mshin77.github.io/TextAnalysisR/articles/python_environment.md) |
