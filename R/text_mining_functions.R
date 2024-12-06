suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(quanteda)
  library(tidytext)
  library(stm)
  library(numform)
  library(textmineR)
  library(widyr)
  library(ggraph)
  library(igraph)
})

#' @title How to Use Functions in TextAnalysisR
#'
#' @description
#' This documentation provides guidance on how to use a set of functions related to:
#' - Word-topic probabilities (Beta)
#' - Document-topic probabilities (Gamma)
#' - Estimated effects of covariates on topic prevalence
#' - Network analysis
#'
#' These steps are similar to those demonstrated in the Shiny web app at \code{TextAnalysisR::TextAnalysisR.app()}.
#'
#' @section Word-Topic Probabilities (Beta):
#' 1. Fit an STM model using \code{stm::stm()}.
#' 2. Extract the beta matrix as a tidy data frame with \code{tidytext::tidy(stm_model, matrix = "beta")}.
#' 3. Use \code{\link{examine_top_terms}} to find top words in each topic.
#' 4. Use \code{\link{plot_topic_term}} to visualize top terms per topic.
#'
#' @section Document-Topic Probabilities (Gamma):
#' 1. Extract gamma from \code{tidytext::tidy(stm_model, matrix = "gamma")}.
#' 2. Use \code{\link{topic_probability_plot}} to visualize topic prevalence.
#' 3. Use \code{\link{topic_probability_table}} to obtain a table of top topics.
#'
#' @section Estimated Effects (Categorical and Continuous Variables):
#' 1. Use \code{stm::estimateEffect()} to model how covariates predict topic proportions.
#' 2. Extract effects using \code{stminsights::get_effects()} and visualize them with custom plotting functions.
#'
#' @section Network Analysis:
#' 1. **Hierarchical Clustering**: Use \code{textmineR::CalcHellingerDist()} on \code{stm_model$theta}, then \code{hclust()} and \code{ggdendro::ggdendrogram()} to visualize.
#' 2. **Text Network**: Convert the dfm to a tidy format, use \code{widyr::pairwise_cor()}, filter by correlation, and visualize with \code{igraph} + \code{ggraph}.
#' 3. **Term Frequency Over Time**: Merge dfm counts with document-level variables, group by a continuous variable, and plot changes in term frequency over time with \code{ggplot2}.
#'
#' @return No return value, called for documentation purposes only.
#' @export
#'
#' @examples
#' # This section provides general guidance; no direct runnable code is required.
how_to_use <- function() {
  invisible(NULL)
}


#' @title Preprocess Text Data
#'
#' @description
#' Preprocesses text data by:
#' - Constructing a corpus
#' - Tokenizing text into words
#' - Converting to lowercase
#' - Removing default English stopwords and optional custom stopwords
#' - Specifying a minimum token length.
#'
#' Typically used before constructing a dfm and fitting an STM model.
#'
#' @param data A data frame that contains text data.
#' @param text_field The name of the column containing text data.
#' @param custom_stopwords A character vector of additional stopwords to remove. Default is NULL.
#' @param min_char Minimum length in characters for tokens (default is 2).
#' @param ... Further arguments passed to \code{quanteda::corpus}.
#'
#' @return A \code{quanteda::tokens} object. This object is a list-like structure where each element
#' represents a tokenized version of a single document. The tokens object can be further processed
#' (e.g., converted into a dfm) for text analysis and modeling.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   d <- data.frame(text = c("This is an example.", "Another example of text."))
#'   result_tokens <- preprocess_texts(d, text_field = "text")
#'   result_tokens
#' }
preprocess_texts <-
  function(data, text_field = "united_texts", custom_stopwords = NULL, min_char = 2, ...) {

    corp <- quanteda::corpus(data, text_field = text_field, ...)
    toks <- quanteda::tokens(corp,
                             what = "word",
                             remove_punct = TRUE,
                             remove_symbols = TRUE,
                             remove_numbers = TRUE,
                             remove_url = TRUE,
                             remove_separators = TRUE,
                             split_hyphens = TRUE,
                             split_tags = TRUE,
                             include_docvars = TRUE,
                             padding = FALSE,
                             verbose = FALSE)
    toks_lower <- quanteda::tokens_tolower(toks, keep_acronyms = FALSE)

    all_stopwords <- c(quanteda::stopwords("en"), custom_stopwords)
    toks_lower_no_stop <- quanteda::tokens_remove(toks_lower,
                                                  all_stopwords,
                                                  valuetype = "glob",
                                                  window = 0,
                                                  verbose = FALSE,
                                                  padding = TRUE)

    toks_clean <- quanteda::tokens_select(toks_lower_no_stop,
                                          min_nchar = min_char,
                                          verbose = FALSE)

    return(toks_clean)
  }


#' @title Plot Word Frequency Results
#'
#' @description
#' Given a document-feature matrix (dfm), this function computes the most frequent terms
#' and creates a ggplot-based visualization of term frequencies.
#'
#' @param data A \code{quanteda} dfm object.
#' @param n The number of top features (terms or words) to display.
#' @param ... Further arguments passed to \code{quanteda.textstats::textstat_frequency}.
#'
#' @return A \code{ggplot} object visualizing the top terms by their frequency. The plot
#' shows each term on one axis and frequency on the other, with points representing their
#' observed frequencies.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   d <- data.frame(text = c("This is an example.", "Another short example text."))
#'   dfm_obj <- d %>%
#'     preprocess_texts(text_field = "text") %>%
#'     quanteda::dfm()
#'   p <- plot_word_frequency(dfm_obj, n = 10)
#'   print(p)
#' }
plot_word_frequency <-
  function(data, n = 20, ...) {
    word_freq <- quanteda.textstats::textstat_frequency(data, n = n, ...)
    word_frequency_plot <- ggplot(word_freq, aes(x = reorder(feature, frequency), y = frequency)) +
      geom_point(colour = "#5f7994", size = 1) +
      coord_flip() +
      labs(x = NULL, y = "Word frequency") +
      theme_minimal(base_size = 10) +
      theme(
        legend.position = "none",
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
        axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
        strip.text.x = element_text(size = 10, color = "#3B3B3B"),
        axis.text.x = element_text(size = 10, color = "#3B3B3B"),
        axis.text.y = element_text(size = 10, color = "#3B3B3B"),
        axis.title = element_text(size = 10, color = "#3B3B3B"),
        axis.title.x = element_text(margin = margin(t = 9)),
        axis.title.y = element_text(margin = margin(r = 9))
      )
    return(word_frequency_plot)
  }


#' @title Examine Highest Per-term Per-topic Probabilities
#'
#' @description
#' Given a tidy data frame of word-topic probabilities (beta values) from an STM model,
#' this function extracts the top terms for each topic.
#'
#' @param data A tidy data frame from \code{tidytext::tidy(stm_model, matrix = "beta")}.
#' @param top_n The number of top terms per topic to return.
#' @param ... Further arguments passed to \code{dplyr::group_by}.
#'
#' @return A \code{tibble} containing the top \code{top_n} terms for each topic. The output includes
#' columns for \code{topic}, \code{term}, and \code{beta} values, restricted to the highest-probability terms.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   # Assume stm_model is a fitted STM object
#'   # beta_td <- tidytext::tidy(stm_model, matrix="beta")
#'   # top_terms <- examine_top_terms(beta_td, top_n = 5)
#'   # head(top_terms)
#' }
examine_top_terms <-
  function(data, top_n, ...) {
    topic_term <- data %>%
      group_by(topic, ...) %>%
      top_n(top_n, beta) %>%
      ungroup()
    return(topic_term)
  }


#' @title Plot Topic Per-term Per-topic Probabilities
#'
#' @description
#' Given per-term per-topic probabilities (beta), this function creates a plot of the top terms in each topic.
#'
#' @param data A tidy data frame from \code{tidytext::tidy(stm_model, matrix="beta")}.
#' @param ncol The number of columns in the facet plot.
#' @param topic_names An optional character vector for labeling topics. If provided, must be the same length as the number of topics.
#' @param ... Further arguments passed to \code{dplyr::group_by}.
#'
#' @return A \code{ggplot} object showing a facet-wrapped chart of top terms for each topic,
#' ordered by their per-topic probability (beta). Each facet represents a topic.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   # Assume beta_td is a tidy data frame of per-term per-topic probabilities
#'   # p <- plot_topic_term(beta_td, ncol = 3)
#'   # print(p)
#' }
plot_topic_term <-
  function(data, ncol = 3, topic_names = NULL, ...) {

    topic_term <- data %>%
      mutate(
        ord = factor(topic, levels = c(min(topic): max(topic))),
        tt = as.numeric(topic),
        topic = paste("Topic", topic),
        term = tidytext::reorder_within(term, beta, topic)
      ) %>%
      arrange(ord)

    levelt = paste("Topic", topic_term$ord) %>% unique()
    topic_term$topic = factor(topic_term$topic, levels = levelt)

    if(!is.null(topic_names)){
      topic_term$topic = topic_names[topic_term$tt]
      topic_term <- topic_term %>%
        mutate(topic = as.character(topic)) %>%
        mutate(topic = ifelse(!is.na(topic), topic, paste('Topic', tt)))
      topic_term$topic = factor(topic_term$topic, levels = unique(topic_term$topic))
    }

    topic_term$tt = NULL

    ggplot(topic_term, aes(term, beta, fill = topic)) +
      geom_col(show.legend = FALSE, alpha = 0.8) +
      facet_wrap(~ topic, scales = "free", ncol = ncol) +
      tidytext::scale_x_reordered() +
      scale_y_continuous(labels = ff_num(zero = 0, digits = 3)) +
      coord_flip() +
      xlab("") +
      ylab("Word probability") +
      theme_minimal(base_size = 10) +
      theme(
        legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
        axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
        strip.text.x = element_text(size = 10, color = "#3B3B3B"),
        axis.text.x = element_text(size = 10, color = "#3B3B3B"),
        axis.text.y = element_text(size = 10, color = "#3B3B3B"),
        axis.title = element_text(size = 10, color = "#3B3B3B"),
        axis.title.x = element_text(margin = margin(t = 9)),
        axis.title.y = element_text(margin = margin(r = 9))
      )
  }


#' @title Plot Per-document Per-topic Probabilities
#'
#' @description
#' Given a tidy data frame of per-document per-topic probabilities (gamma),
#' this function calculates the mean topic prevalence across documents and plots the top topics.
#'
#' @param data A tidy data frame from \code{tidytext::tidy(stm_model, matrix="gamma")}.
#' @param top_n The number of topics to display, ordered by their mean prevalence.
#'
#' @return A \code{ggplot} object showing a bar plot of topic prevalence. Topics are ordered by their
#' mean gamma value (average prevalence across documents).
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   # Assume gamma_td is from tidytext::tidy(stm_model, matrix="gamma")
#'   # p <- topic_probability_plot(gamma_td, top_n = 10)
#'   # print(p)
#' }
topic_probability_plot <-
  function(data, top_n = 10) {

    gamma_terms <- data %>%
      group_by(topic) %>%
      summarise(gamma = mean(gamma)) %>%
      arrange(desc(gamma)) %>%
      mutate(topic = reorder(topic, gamma)) %>%
      top_n(top_n, gamma)

    ggplot(gamma_terms, aes(topic, gamma, fill = topic)) +
      geom_col(alpha = 0.8) +
      coord_flip() +
      scale_y_continuous(labels = ff_num(zero = 0, digits = 2)) +
      xlab("") +
      ylab("Topic proportion") +
      theme_minimal(base_size = 10) +
      theme(
        legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
        axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
        strip.text.x = element_text(size = 10, color = "#3B3B3B"),
        axis.text.x = element_text(size = 10, color = "#3B3B3B"),
        axis.text.y = element_text(size = 10, color = "#3B3B3B"),
        axis.title = element_text(size = 10, color = "#3B3B3B"),
        axis.title.x = element_text(margin = margin(t = 9)),
        axis.title.y = element_text(margin = margin(r = 9))
      )
  }


#' @title Create a Table of Topic Prevalence
#'
#' @description
#' Given a tidy data frame of per-document per-topic probabilities (gamma),
#' this function calculates the mean prevalence of each topic and returns a table of the top topics.
#'
#' @param data A tidy data frame from \code{tidytext::tidy(stm_model, matrix="gamma")}.
#' @param top_n The number of topics to display, ordered by their mean prevalence.
#'
#' @return A \code{tibble} containing columns \code{topic} and \code{gamma}, where \code{topic}
#' is a factor representing each topic (relabeled with a "Topic X" format), and \code{gamma} is the
#' mean topic prevalence across all documents. Numeric values are rounded to three decimal places.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   # Assume gamma_td is from tidytext::tidy(stm_model, matrix="gamma")
#'   # tbl <- topic_probability_table(gamma_td, top_n = 10)
#'   # print(tbl)
#' }
topic_probability_table <-
  function(data, top_n = 10) {

    gamma_terms <- data %>%
      group_by(topic) %>%
      summarise(gamma = mean(gamma)) %>%
      arrange(desc(gamma)) %>%
      mutate(topic = reorder(topic, gamma)) %>%
      top_n(top_n, gamma) %>%
      mutate(tt = as.numeric(topic)) %>%
      mutate(ord = topic) %>%
      mutate(topic = paste('Topic', topic)) %>%
      arrange(ord)

    levelt = paste("Topic", gamma_terms$ord) %>% unique()
    gamma_terms$topic = factor(gamma_terms$topic, levels = levelt)

    gamma_terms %>%
      select(topic, gamma) %>%
      mutate_if(is.numeric, ~ round(., 3))
  }

