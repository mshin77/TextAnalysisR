% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/text_mining_functions.R
\name{examine_top_terms}
\alias{examine_top_terms}
\title{Examine highest word probabilities for each topic}
\usage{
examine_top_terms(data, top_n, ...)
}
\arguments{
\item{data}{A tidy data frame that includes term-topic probabilities
(probabilities of each word per each topic).}

\item{top_n}{A number of top n terms with highest term-topic probabilities in each document.}

\item{...}{Further arguments passed to \code{dplyr::group_by}.}
}
\value{
A tibble (data frame) object with a list of word probabilities from \code{tidytext::tidy}.
The result is a data frame containing word probabilities for each topic.
}
\description{
Examine highest document-topic probabilities.
}
\examples{
if(requireNamespace("quanteda", "tidytext")){
dfm <- SpecialEduTech \%>\% preprocess_texts(text_field = "abstract") \%>\% quanteda::dfm()
data <- tidytext::tidy(stm_15, document_names = rownames(dfm), log = FALSE)
data \%>\% examine_top_terms(top_n = 5)
}

}
