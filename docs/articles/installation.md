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
# Python spaCy required - see setup_python_env()
TextAnalysisR::setup_python_env()
```

### Python Features

For PDF tables, embeddings, and topic-grounded analysis:

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

## Responsible AI Design

TextAnalysisR follows responsible AI principles with human oversight:

- **AI Suggests**: LLMs generate draft labels, content, and
  recommendations
- **Human Reviews**: You examine all AI outputs before use
- **Human Decides**: Edit, approve, or regenerate as needed
- **Human Controls**: Override any AI suggestion with manual input

This approach aligns with [NIST AI Risk Management
Framework](https://www.nist.gov/itl/ai-risk-management-framework) and EU
AI Act requirements for meaningful human control.
