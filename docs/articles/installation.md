# Installation

``` r

library(TextAnalysisR)
packageVersion("TextAnalysisR")
```

    ## [1] '0.1.4'

``` r

data("SpecialEduTech")
nrow(SpecialEduTech)
```

    ## [1] 490

``` r

head(SpecialEduTech[, c("title", "year")], 3)
```

    ## # A tibble: 3 × 2
    ##   title                                                                     year
    ##   <chr>                                                                    <dbl>
    ## 1 Dyscalculia and the minicalculator: the ALP program                       1980
    ## 2 The effects of computer-assisted instruction for mastery of multiplicat…  1981
    ## 3 Computer Assisted Instruction with Learning Disabled Students             1981

## Choose a Setup

| Goal | Setup |
|----|----|
| Small datasets in the browser | Web app at [textanalysisr.org](https://www.textanalysisr.org) |
| Full analyses and large datasets | [`install.packages()`](https://rdrr.io/r/utils/install.packages.html) below |
| spaCy annotation and local transformer models | [`setup_python_env()`](https://mshin77.github.io/TextAnalysisR/reference/setup_python_env.md) after installing |

Python is optional. Core analyses run in plain R; cloud AI providers
(OpenAI, Gemini) can substitute for local models with an API key.

## R Package

``` r

install.packages("TextAnalysisR",
  repos = c("https://mshin77.r-universe.dev", "https://cloud.r-project.org"))
library(TextAnalysisR)
run_app()
```

Requirements: R \>= 4.0, RStudio recommended

## Web App

Visit [textanalysisr.org](https://www.textanalysisr.org) - no
installation needed.

Note: Web version has limited features (no Python, no large files).
Cloud AI providers (OpenAI, Gemini) are available with an API key.

## Optional Features

### Linguistic Analysis (spaCy)

For lemmatization, POS tagging, and named entity recognition, set up
Python with
[`TextAnalysisR::setup_python_env()`](https://mshin77.github.io/TextAnalysisR/reference/setup_python_env.md).

### Python Features

For embeddings and neural sentiment analysis, run
[`setup_python_env()`](https://mshin77.github.io/TextAnalysisR/reference/setup_python_env.md).
Requires Python 3.9+.

## Troubleshooting

| Issue | Solution |
|----|----|
| Package install failed | `remotes::install_github("mshin77/TextAnalysisR", dependencies = TRUE)` |
| Browser doesn’t open | Navigate to URL shown in R console |
| Python errors | See [Python Environment](https://mshin77.github.io/TextAnalysisR/articles/python_environment.md) |
