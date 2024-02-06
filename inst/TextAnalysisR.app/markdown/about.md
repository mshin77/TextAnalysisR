<!-- README.md is generated from README.Rmd. Please edit that file -->
<!-- badges: start -->

[![R-CMD-check](https://github.com/mshin77/TextAnalysisR/workflows/R-CMD-check/badge.svg)](https://github.com/mshin77/TextAnalysisR/actions)

`TextAnalysisR` provides a supporting workflow for text mining analysis.
The web app incorporates
[quanteda](https://github.com/quanteda/quanteda) (text preprocessing),
[stm](https://github.com/bstewart/stm) (structural topic modeling), and
[ggraph](https://github.com/thomasp85/ggraph) as well as
[widyr](https://github.com/juliasilge/widyr) (network analysis).
[tidytext](https://github.com/cran/tidytext) was implemented to tidy
non-tidy format objects.

## Installation

The development version from [GitHub](https://github.com/) with:

    install.packages("devtools")
    devtools::install_github("mshin77/TextAnalysisR")

## Example

Launch and browser the TextAnalysisR app:

    library(TextAnalysisR)
    TextAnalysisR.app()
