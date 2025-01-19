
#' @title Unite Text Columns
#'
#' @description
#' This function unites specified text columns in a data frame into a single column named "united_texts" while retaining the original columns.
#'
#' @param df A data frame that contains text data.
#' @param listed_vars A character vector of column names to be united into "united_texts".
#'
#' @return A data frame with a new column "united_texts" created by uniting the specified variables.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   united_tbl
#' }
unite_text_cols <- function(df, listed_vars) {
  united_texts_tbl <- df %>%
    dplyr::select(all_of(unname(listed_vars))) %>%
    tidyr::unite(col = "united_texts", sep = " ", remove = TRUE)

  docvar_tbl <- df

  united_tbl <- dplyr::bind_cols(united_texts_tbl, docvar_tbl)

  return(united_tbl)
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
#' @param united_tbl A data frame that contains text data.
#' @param text_field The name of the column in \code{united_tbl} that contains text data.
#' @param custom_stopwords A character vector of additional stopwords to remove. Default is NULL.
#' @param min_char Minimum length in characters for tokens (default is 2).
#' @param ... Further arguments passed to \code{quanteda::corpus}.
#'
#' @return A \code{quanteda} tokens object. Each row in the object represents a document, and each column represents a token.
#' The object is ready for constructing a dfm and fitting an STM model.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'   tokens
#' }
preprocess_texts <- function(united_tbl, text_field = "united_texts", custom_stopwords = NULL, min_char = 2, ...) {

  corp <- quanteda::corpus(united_tbl, text_field = text_field, ...)

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


#' @title Plot Word Frequency
#'
#' @description
#' Given a document-feature matrix (dfm), this function computes the most frequent terms
#' and creates a ggplot-based visualization of term frequencies.
#'
#' @param dfm_object A \code{quanteda} dfm object.
#' @param n The number of top terms to display (default: 20).
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
#'   df <- TextAnalysisR::SpecialEduTech
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'   dfm_object <- quanteda::dfm(tokens)
#'   word_frequency_plot <- TextAnalysisR::plot_word_frequency(dfm_object, n = 20)
#'   word_frequency_plot
#' }
plot_word_frequency <-
  function(dfm_object, n = 20, ...) {
    word_freq <- quanteda.textstats::textstat_frequency(dfm_object, n = n, ...)
    word_frequency_plot <- ggplot(word_freq, aes(x = reorder(feature, frequency), y = frequency)) +
      geom_point(colour = "#5f7994", size = 1) +
      coord_flip() +
      labs(x = NULL, y = "Word frequency") +
      theme_minimal(base_size = 11) +
      theme(
        legend.position = "none",
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
        axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
        strip.text.x = element_text(size = 11, color = "#3B3B3B"),
        axis.text.x = element_text(size = 11, color = "#3B3B3B"),
        axis.text.y = element_text(size = 11, color = "#3B3B3B"),
        axis.title = element_text(size = 11, color = "#3B3B3B"),
        axis.title.x = element_text(margin = margin(t = 9)),
        axis.title.y = element_text(margin = margin(r = 9))
      )
    return(word_frequency_plot)
  }


#' @title Remove Common Words Across Documents
#'
#' @description
#' This function removes specified common words from a tokens object and applies two dictionaries
#' to categorize the remaining tokens. It returns a document-feature matrix (dfm) based on the
#' processed tokens. If no words are specified for removal, it returns an initial dfm using the
#' provided initialization function.
#'
#' @param tokens A \code{tokens} object from the \code{quanteda} package, typically processed
#'   using functions like \code{tokens_select} or \code{tokens_remove}.
#' @param remove_vars A character vector of words to remove from the tokens. If \code{NULL},
#'   the function returns the result of \code{dfm_init_func()}.
#' @param dfm_object A \code{dfm} object to process after removing the specified words.
#'
#' @return A \code{dfm} object with the specified words removed and the remaining tokens categorized
#'
#' @importFrom quanteda tokens_remove tokens_lookup dfm dictionary
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'   dfm_object <- quanteda::dfm(tokens)
#'   TextAnalysisR::remove_common_words(tokens = tokens,
#'                                      remove_vars = c("level", "testing"),
#'                                      dfm_object = dfm_object)
#' }
remove_common_words <- function(tokens, remove_vars, dfm_object) {

  dictionary_list_1 <- TextAnalysisR::dictionary_list_1
  dictionary_list_2 <- TextAnalysisR::dictionary_list_2

  if (!is.null(remove_vars)) {
    removed_processed_tokens <- quanteda::tokens_remove(tokens, remove_vars)

    removed_tokens_dict_int <- quanteda::tokens_lookup(
      removed_processed_tokens,
      dictionary = dictionary(dictionary_list_1),
      valuetype = "glob",
      verbose = FALSE,
      exclusive = FALSE,
      capkeys = FALSE
    )

    removed_tokens_dict <- quanteda::tokens_lookup(
      removed_tokens_dict_int,
      dictionary = dictionary(dictionary_list_2),
      valuetype = "glob",
      verbose = FALSE,
      exclusive = FALSE,
      capkeys = FALSE
    )

    quanteda::dfm(removed_tokens_dict)

  }
}


#' @title Evaluate Optimal Number of Topics
#'
#' @description
#' This function performs a search for the optimal number of topics (K) using \code{stm::searchK}
#' and visualizes diagnostics, including held-out likelihood, residuals, semantic coherence,
#' and lower bound metrics.
#'
#' @param dfm_object A \code{quanteda} document-feature matrix (dfm).
#' @param topic_range A numeric vector specifying the range of topics (K) to search over.
#' @param max.em.its Maximum number of EM iterations (default: 75).
#' @param categorical_var An optional character string for a categorical variable in the metadata.
#' @param continuous_var An optional character string for a continuous variable in the metadata.
#' @param height The height of the resulting Plotly plot in pixels (default: 600).
#' @param width The width of the resulting Plotly plot in pixels (default: 800).
#' @param verbose Logical; if \code{TRUE}, prints progress information.
#' @param ... Further arguments passed to \code{stm::searchK}.
#'
#' @return A \code{plotly} object showing the diagnostics for the number of topics (K).
#'
#' @importFrom quanteda convert
#' @importFrom stm searchK
#' @importFrom plotly plot_ly subplot layout
#' @importFrom dplyr mutate select
#' @importFrom stats as.formula
#' @importFrom utils str
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'   dfm_object <- quanteda::dfm(tokens)
#'   TextAnalysisR::evaluate_optimal_topic_number(
#'     dfm_object = dfm_object,
#'     topic_range = 5:30,
#'     max.em.its = 75,
#'     categorical_var = "reference_type",
#'     continuous_var = "year",
#'     height = 600,
#'     width = 800,
#'     verbose = TRUE
#'   )
#' }
evaluate_optimal_topic_number <- function(dfm_object,
                                          topic_range,
                                          max.em.its = 75,
                                          categorical_var = NULL,
                                          continuous_var = NULL,
                                          height = 600,
                                          width = 800,
                                          verbose = TRUE, ...) {

  out <- quanteda::convert(dfm_object, to = "stm")

  if (is.null(out$meta) || is.null(out$documents) || is.null(out$vocab)) {
    stop("Conversion to STM format failed. Please ensure your dfm_object is correctly formatted.")
  }

  categorical_var <- if (!is.null(categorical_var)) as.character(categorical_var) else NULL
  continuous_var <- if (!is.null(continuous_var)) as.character(continuous_var) else NULL

  if (!is.null(categorical_var)) {
    categorical_var <- unlist(strsplit(categorical_var, ",\\s*"))
  }
  if (!is.null(continuous_var)) {
    continuous_var <- unlist(strsplit(continuous_var, ",\\s*"))
  }

  missing_vars <- setdiff(c(categorical_var, continuous_var), names(out$meta))
  if (length(missing_vars) > 0) {
    stop("The following variables are missing in the metadata: ", paste(missing_vars, collapse = ", "))
  }

  terms <- c()
  if (!is.null(categorical_var) && length(categorical_var) > 0) {
    terms <- c(terms, categorical_var)
  }
  if (!is.null(continuous_var) && length(continuous_var) > 0) {
    terms <- c(terms, continuous_var)
  }

  prevalence_formula <- if (length(terms) > 0) {
    as.formula(paste("~", paste(terms, collapse = " + ")))
  } else {
    NULL
  }

  search_result <- tryCatch({
    stm::searchK(
      data = out$meta,
      documents = out$documents,
      vocab = out$vocab,
      max.em.its = max.em.its,
      init.type = "Spectral",
      K = topic_range,
      prevalence = prevalence_formula,
      verbose = verbose,
      ...
    )
  }, error = function(e) {
    stop("Error in stm::searchK: ", e$message)
  })

  # print(search_result$results)

  search_result$results$heldout <- as.numeric(search_result$results$heldout)
  search_result$results$residual <- as.numeric(search_result$results$residual)
  search_result$results$semcoh <- as.numeric(search_result$results$semcoh)
  search_result$results$lbound <- as.numeric(search_result$results$lbound)

  p1 <- plotly::plot_ly(
    data = search_result$results,
    x = ~K,
    y = ~heldout,
    type = 'scatter',
    mode = 'lines+markers',
    text = ~paste("K:", K, "<br>Held-out Likelihood:", round(heldout, 3)),
    hoverinfo = 'text',
    width = width,
    height = height
  )

  p2 <- plotly::plot_ly(
    data = search_result$results,
    x = ~K,
    y = ~residual,
    type = 'scatter',
    mode = 'lines+markers',
    text = ~paste("K:", K, "<br>Residuals:", round(residual, 3)),
    hoverinfo = 'text',
    width = width,
    height = height
  )

  p3 <- plotly::plot_ly(
    data = search_result$results,
    x = ~K,
    y = ~semcoh,
    type = 'scatter',
    mode = 'lines+markers',
    text = ~paste("K:", K, "<br>Semantic Coherence:", round(semcoh, 3)),
    hoverinfo = 'text',
    width = width,
    height = height
  )

  p4 <- plotly::plot_ly(
    data = search_result$results,
    x = ~K,
    y = ~lbound,
    type = 'scatter',
    mode = 'lines+markers',
    text = ~paste("K:", K, "<br>Lower Bound:", round(lbound, 3)),
    hoverinfo = 'text',
    width = width,
    height = height
  )

  plotly::subplot(p1, p2, p3, p4, nrows = 2, margin = 0.1) %>%
    plotly::layout(
      title = list(
        text = "Model Diagnostics by Number of Topics (K)",
        font = list(size = 16)
      ),
      showlegend = FALSE,
      margin = list(t = 100, b = 150, l = 50, r = 50),
      annotations = list(
        list(
          x = 0.25, y = 1.05, text = "Held-out Likelihood", showarrow = FALSE,
          xref = 'paper', yref = 'paper', xanchor = 'center', yanchor = 'bottom',
          font = list(size = 14)
        ),
        list(
          x = 0.75, y = 1.05, text = "Residuals", showarrow = FALSE,
          xref = 'paper', yref = 'paper', xanchor = 'center', yanchor = 'bottom',
          font = list(size = 14)
        ),
        list(
          x = 0.25, y = 0.5, text = "Semantic Coherence", showarrow = FALSE,
          xref = 'paper', yref = 'paper', xanchor = 'center', yanchor = 'bottom',
          font = list(size = 14)
        ),
        list(
          x = 0.75, y = 0.5, text = "Lower Bound", showarrow = FALSE,
          xref = 'paper', yref = 'paper', xanchor = 'center', yanchor = 'bottom',
          font = list(size = 14)
        ),
        list(
          x = 0.5, y = -0.2, text = "Number of Topics (K)", showarrow = FALSE,
          xref = 'paper', yref = 'paper', xanchor = 'center', yanchor = 'top',
          font = list(size = 14)
        )
      ),
      yaxis = list(
        title = list(
          text = "Metric Value",
          font = list(size = 14)
        )
      )
    )
}


#' @title Plot Highest Word Probabilities for Each Topic
#'
#' @description
#' This function provides a visualization of the top terms for each topic,
#' ordered by their word probability distribution for each topic (beta).
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param topic_n The number of topics to display.
#' @param max.em.its Maximum number of EM iterations (default: 75).
#' @param categorical_var An optional character string for a categorical variable in the metadata.
#' @param continuous_var An optional character string for a continuous variable in the metadata.
#' @param top_term_n The number of top terms to display for each topic (default: 10).
#' @param ncol The number of columns in the facet plot (default: 3).
#' @param topic_names An optional character vector for labeling topics. If provided, must be the same length as the number of topics.
#' @param height The height of the resulting Plotly plot, in pixels. Defaults to \code{1200}.
#' @param width The width of the resulting Plotly plot, in pixels. Defaults to \code{800}.
#' @param verbose Logical; if \code{TRUE}, prints progress information.
#' @param ... Further arguments passed to \code{stm::searchK}.
#'
#' @return A \code{Plotly} object showing a facet-wrapped chart of top terms for each topic,
#' ordered by their per-topic probability (beta). Each facet represents a topic.
#'
#' @details
#' If \code{topic_names} is provided, it replaces the default "Topic \{n\}" labels with custom names.
#'
#' @importFrom stats reorder
#' @importFrom numform ff_num
#' @importFrom plotly ggplotly layout
#' @importFrom tidytext reorder_within scale_x_reordered
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'   dfm_object <- quanteda::dfm(tokens)
#' TextAnalysisR::plot_word_probabilities(
#'   dfm_object = dfm_object,
#'   topic_n = 15,
#'   max.em.its = 75,
#'   categorical_var = "reference_type",
#'   continuous_var = "year",
#'   top_term_n = 10,
#'   ncol = 3,
#'   height = 1200,
#'   width = 800,
#'   verbose = TRUE)
#' }
plot_word_probabilities <- function(dfm_object,
                                    topic_n,
                                    max.em.its = 75,
                                    categorical_var = NULL,
                                    continuous_var = NULL,
                                    top_term_n = 10,
                                    ncol = 3,
                                    topic_names = NULL,
                                    height = 1200,
                                    width = 800,
                                    verbose = TRUE, ...) {

  out <- quanteda::convert(dfm_object, to = "stm")
  if (!all(c("meta", "documents", "vocab") %in% names(out))) {
    stop("Conversion of dfm_outcome must result in 'meta', 'documents', and 'vocab'.")
  }

  meta <- out$meta
  documents <- out$documents
  vocab <- out$vocab

  prevalence_formula <- NULL

  if (!is.null(categorical_var) && !is.null(continuous_var)) {
    prevalence_formula <- reformulate(c(categorical_var, sprintf("s(%s)", continuous_var)))
  } else if (!is.null(categorical_var)) {
    prevalence_formula <- reformulate(categorical_var)
  } else if (!is.null(continuous_var)) {
    prevalence_formula <- reformulate(sprintf("s(%s)", continuous_var))
  }

  stm_model <- stm::stm(
    data = meta,
    documents = documents,
    vocab = vocab,
    K = topic_n,
    prevalence = prevalence_formula,
    max.em.its = max.em.its,
    init.type = "Spectral",
    verbose = verbose,
    ...
  )

  beta_td <- tidytext::tidy(stm_model, matrix = "beta")

  topic_term_plot <- beta_td %>%
    dplyr::group_by(topic) %>%
    slice_max(order_by = beta, n = top_term_n) %>%
    ungroup() %>%
    mutate(
      ord = factor(topic, levels = c(min(topic):max(topic))),
      tt = as.numeric(topic),
      topic = paste("Topic", topic),
      term = reorder_within(term, beta, topic)
    ) %>%
    arrange(ord) %>%
    ungroup()

  levelt = paste("Topic", topic_term_plot$ord) %>% unique()
  topic_term_plot$topic = factor(topic_term_plot$topic,
                                 levels = levelt)

  topic_term_plot_gg <- ggplot(
    topic_term_plot,
    aes(term, beta, fill = topic, text = paste("Topic:", topic, "<br>Beta:", sprintf("%.3f", beta)))
  ) +
    geom_col(show.legend = FALSE, alpha = 0.9) +
    facet_wrap(~ topic, scales = "free", ncol = ncol, strip.position = "top") +
    scale_x_reordered() +
    scale_y_continuous(labels = numform::ff_num(zero = 0, digits = 3)) +
    coord_flip() +
    xlab("") +
    ylab("Word probability") +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "none",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
      strip.text.x = element_text(size = 11, color = "#3B3B3B", margin = margin(b = 30, t = 15)),
      axis.text.x = element_text(size = 11, color = "#3B3B3B", hjust = 1, margin = margin(t = 20)),
      axis.text.y = element_text(size = 11, color = "#3B3B3B", margin = margin(r = 20)),
      axis.title = element_text(size = 11, color = "#3B3B3B"),
      axis.title.x = element_text(margin = margin(t = 25)),
      axis.title.y = element_text(margin = margin(r = 25)),
      plot.margin = margin(t = 40, b = 40)
    )

  plotly::ggplotly(topic_term_plot_gg, height = height, width = width, tooltip = "text") %>%
    plotly::layout(
      margin = list(t = 40, b = 40)
    )
}


#' @title Plot Mean Topic Prevalence Across Documents
#'
#' @description
#' This function calculates the mean topic prevalence across documents and plots the top topics.
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param topic_n The number of topics to display.
#' @param max.em.its Maximum number of EM iterations (default: 75).
#' @param categorical_var An optional character string for a categorical variable in the metadata.
#' @param continuous_var An optional character string for a continuous variable in the metadata.
#' @param top_term_n The number of top terms to display for each topic (default: 10).
#' @param top_topic_n The number of top topics to display (default: 15).
#' @param topic_names An optional character vector for labeling topics. If provided, must be the same length as the number of topics.
#' @param height The height of the resulting Plotly plot, in pixels. Defaults to \code{500}.
#' @param width The width of the resulting Plotly plot, in pixels. Defaults to \code{1000}.
#' @param verbose Logical; if \code{TRUE}, prints progress information (default: FALSE).
#' @param ... Further arguments passed to \code{stm::searchK}.
#'
#' @return A \code{ggplot} object showing a bar plot of topic prevalence. Topics are ordered by their
#' mean gamma value (average prevalence across documents).
#'
#' @details
#' If \code{topic_names} is provided, it replaces the default "Topic \{n\}" labels with custom names.#'
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'   dfm_object <- quanteda::dfm(tokens)
#' TextAnalysisR::plot_mean_topic_prevalence(
#'   dfm_object = dfm_object,
#'   topic_n = 15,
#'   max.em.its = 75,
#'   categorical_var = "reference_type",
#'   continuous_var = "year",
#'   top_term_n = 10,
#'   top_topic_n = 15,
#'   height = 500,
#'   width = 900,
#'   verbose = TRUE)
#' }
#'
#' @importFrom stats reorder
#' @importFrom numform ff_num
#' @importFrom plotly ggplotly layout
#' @importFrom tidytext reorder_within scale_x_reordered
plot_mean_topic_prevalence <- function(dfm_object,
                                       topic_n,
                                       max.em.its = 75,
                                       categorical_var = NULL,
                                       continuous_var = NULL,
                                       top_term_n = 10,
                                       top_topic_n = 15,
                                       topic_names = NULL,
                                       height = 500,
                                       width = 900,
                                       verbose = TRUE, ...) {

  out <- quanteda::convert(dfm_object, to = "stm")
  if (!all(c("meta", "documents", "vocab") %in% names(out))) {
    stop("Conversion of dfm_object must result in 'meta', 'documents', and 'vocab'.")
  }

  meta <- out$meta
  documents <- out$documents
  vocab <- out$vocab

  prevalence_formula <- NULL

  if (!is.null(categorical_var) && !is.null(continuous_var)) {
    prevalence_formula <- reformulate(c(categorical_var, sprintf("s(%s)", continuous_var)))
  } else if (!is.null(categorical_var)) {
    prevalence_formula <- reformulate(categorical_var)
  } else if (!is.null(continuous_var)) {
    prevalence_formula <- reformulate(sprintf("s(%s)", continuous_var))
  }

  stm_model <- stm::stm(
    data = meta,
    documents = documents,
    vocab = vocab,
    K = topic_n,
    prevalence = prevalence_formula,
    max.em.its = max.em.its,
    init.type = "Spectral",
    verbose = verbose,
    ...
  )

  beta_td <- tidytext::tidy(stm_model, matrix = "beta")

  top_terms_selected <- beta_td %>%
    arrange(beta) %>%
    group_by(topic) %>%
    top_n(top_term_n, beta) %>%
    arrange(beta) %>%
    select(topic, term) %>%
    summarise(terms = list(term)) %>%
    mutate(terms = purrr::map(terms, paste, collapse = ", ")) %>%
    unnest(cols = c(terms))

  gamma_td <- tidytext::tidy(stm_model, matrix = "gamma", document_names = rownames(dfm_object))

  gamma_terms <- gamma_td %>%
    group_by(topic) %>%
    summarise(gamma = mean(gamma)) %>%
    arrange(desc(gamma)) %>%
    mutate(topic = as.integer(as.character(topic))) %>%
    left_join(top_terms_selected, by = "topic") %>%
    mutate(topic = reorder(topic, gamma))

  topic_by_prevalence_plot <- gamma_terms %>%
    top_n(top_topic_n, gamma) %>%
    mutate(tt = as.numeric(topic)) %>%
    mutate(ord = topic) %>%
    mutate(topic = paste('Topic', topic)) %>%
    arrange(ord)

  levelt <- paste("Topic", topic_by_prevalence_plot$ord) %>% unique()

  topic_by_prevalence_plot$topic <- factor(topic_by_prevalence_plot$topic,
                                           levels = levelt)

  tp <- topic_by_prevalence_plot %>%
    ggplot(aes(topic, gamma,
               label = terms,
               fill = topic,
               text = paste("Topic:", topic, "<br>Terms:",
                            terms, "<br>Gamma:", sprintf("%.3f", gamma)))
    ) +
    geom_col(alpha = 0.9) +
    coord_flip() +
    scale_y_continuous(labels = numform::ff_num(zero = 0, digits = 2)) +
    xlab(" ") +
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

  tp %>% plotly::ggplotly(tooltip = "text") %>%
    plotly::layout(
      margin = list(t = 40, b = 40)
    )
}

#' @title Analyze and Visualize Word Co-occurrence Networks
#'
#' @description
#' This function creates a word co-occurrence network based on a document-feature matrix (dfm).
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param co_occur_n Minimum number of co-occurrences for filtering terms (default is 130).
#' @param top_node_n Number of top nodes to display (default is 30).
#' @param height The height of the resulting Plotly plot, in pixels. Defaults to \code{800}.
#' @param width The width of the resulting Plotly plot, in pixels. Defaults to \code{900}.
#'
#' @return A list containing a Plotly object and a data frame with the results.
#'
#' @importFrom igraph graph_from_data_frame V vcount degree betweenness closeness eigen_centrality layout_with_fr
#' @importFrom plotly plot_ly add_segments add_markers layout
#' @importFrom dplyr count filter mutate select group_by summarise
#' @importFrom tibble as_tibble
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_count
#' @importFrom scales rescale
#' @importFrom stats quantile
#' @importFrom DT datatable
#' @importFrom shiny showNotification
#' @importFrom rlang sym
#' @importFrom utils head
#' @importFrom grDevices colorRampPalette
#' @importFrom RColorBrewer brewer.pal
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'   dfm_object <- quanteda::dfm(tokens)
#'   word_co_occurrence_network_results <- TextAnalysisR::word_co_occurrence_network(
#'                                         dfm_object,
#'                                         co_occur_n = 130,
#'                                         top_node_n = 30,
#'                                         height = 800,
#'                                         width = 900)
#'   word_co_occurrence_network_results$plot
#'   word_co_occurrence_network_results$table
#'   word_co_occurrence_network_results$summary
#' }
word_co_occurrence_network <- function(dfm_object,
                                       co_occur_n = 130,
                                       top_node_n = 30,
                                       height = 800,
                                       width = 900) {

  dfm_td <- tidytext::tidy(dfm_object) %>%
    tibble::as_tibble()

  term_co_occur <- dfm_td %>%
    count(document, term) %>%
    widyr::pairwise_count(term, document, sort = TRUE) %>%
    filter(n >= co_occur_n)

  co_occur_graph <- igraph::graph_from_data_frame(term_co_occur, directed = FALSE)

  if (igraph::vcount(co_occur_graph) == 0) {
    showNotification("No co-occurrence relationships meet the threshold.", type = "error")
    return(NULL)
  }

  igraph::V(co_occur_graph)$degree <- igraph::degree(co_occur_graph)
  igraph::V(co_occur_graph)$betweenness <- igraph::betweenness(co_occur_graph)
  igraph::V(co_occur_graph)$closeness <- igraph::closeness(co_occur_graph)
  igraph::V(co_occur_graph)$eigenvector <- igraph::eigen_centrality(co_occur_graph)$vector
  igraph::V(co_occur_graph)$community <- igraph::cluster_leiden(co_occur_graph)$membership

  layout <- igraph::layout_with_fr(co_occur_graph)
  layout_df <- as.data.frame(layout)
  colnames(layout_df) <- c("x", "y")

  layout_df$label <- igraph::V(co_occur_graph)$name
  layout_df$degree <- igraph::V(co_occur_graph)$degree
  layout_df$betweenness <- igraph::V(co_occur_graph)$betweenness
  layout_df$closeness <- igraph::V(co_occur_graph)$closeness
  layout_df$eigenvector <- igraph::V(co_occur_graph)$eigenvector
  layout_df$community <- igraph::V(co_occur_graph)$community

  edge_data <- igraph::as_data_frame(co_occur_graph, what = "edges") %>%
    dplyr::mutate(
      x = layout_df$x[match(from, layout_df$label)],
      y = layout_df$y[match(from, layout_df$label)],
      xend = layout_df$x[match(to, layout_df$label)],
      yend = layout_df$y[match(to, layout_df$label)],
      cooccur_count = n
    ) %>%
    dplyr::select(from, to, x, y, xend, yend, cooccur_count)

  edge_data <- edge_data %>%
    dplyr::mutate(
      line_group = as.integer(cut(
        cooccur_count,
        breaks = unique(quantile(cooccur_count, probs = seq(0, 1, length.out = 6), na.rm = TRUE)),
        include.lowest = TRUE
      )),
      line_width = scales::rescale(line_group, to = c(1, 5)),
      alpha = scales::rescale(line_group, to = c(0.1, 0.3))
    )

  edge_group_labels <- edge_data %>%
    dplyr::group_by(line_group) %>%
    dplyr::summarise(
      min_count = min(cooccur_count, na.rm = TRUE),
      max_count = max(cooccur_count, na.rm = TRUE)
    ) %>%
    dplyr::mutate(label = paste0("Count: ", min_count, " - ", max_count)) %>%
    dplyr::pull(label)

  node_data <- layout_df %>%
    dplyr::mutate(
      degree_log = log1p(degree),
      size = scales::rescale(degree_log, to = c(12, 30)),
      text_size = scales::rescale(degree_log, to = c(14, 20)),
      alpha = scales::rescale(degree_log, to = c(0.2, 1)),
      hover_text = paste(
        "Word:", label,
        "<br>Degree:", degree,
        "<br>Betweenness:", round(betweenness, 2),
        "<br>Closeness:", round(closeness, 2),
        "<br>Eigenvector:", round(eigenvector, 2),
        "<br>Community:", community
      )
    )

  n_communities <- length(unique(node_data$community))
  if (n_communities >= 3 && n_communities <= 8) {
    palette <- RColorBrewer::brewer.pal(n_communities, "Set2")
  } else if (n_communities > 8) {
    palette <- colorRampPalette(RColorBrewer::brewer.pal(8, "Set2"))(n_communities)
  } else if (n_communities > 0 && n_communities < 3) {
    palette <- RColorBrewer::brewer.pal(3, "Set2")[1:n_communities]
  } else {
    palette <- rep("#000000", n_communities)
  }

  node_data$community <- factor(node_data$community, levels = unique(node_data$community))
  community_levels <- levels(node_data$community)
  names(palette) <- community_levels
  node_data$color <- palette[as.character(node_data$community)]

  plot <- plotly::plot_ly(
    type = 'scatter',
    mode = 'markers',
    width = width,
    height = height
  )

  for (i in unique(edge_data$line_group)) {

    edge_subset <- edge_data %>% dplyr::filter(line_group == i)
    edge_label <- edge_group_labels[i]

    edge_subset <- edge_subset %>%
      dplyr::mutate(
        mid_x = (x + xend) / 2,
        mid_y = (y + yend) / 2
      )

    if (nrow(edge_subset) > 0) {
      plot <- plot %>%
        plotly::add_segments(
          data = edge_subset,
          x = ~x,
          y = ~y,
          xend = ~xend,
          yend = ~yend,
          line = list(
            color = '#5C5CFF',
            width = ~line_width
          ),
          hoverinfo = 'none',
          opacity = ~alpha,
          showlegend = TRUE,
          name = edge_label,
          legendgroup = "Edges"
        ) %>%

        plotly::add_trace(
          data = edge_subset,
          x = ~mid_x,
          y = ~mid_y,
          type = 'scatter',
          mode = 'markers',
          marker = list(size = 0.1, color = '#e0f7ff', opacity = 0),
          text = ~paste0(
            "Co-occurrence: ", cooccur_count,
            "<br>Source: ", from,
            "<br>Target: ", to
          ),
          hoverinfo = 'text',
          showlegend = FALSE
        )
    }
  }

  plot <- plot %>%
    plotly::layout(
      legend = list(
        title = list(text = "Co-occurrence"),
        orientation = "v",
        x = 1.1,
        y = 1,
        xanchor = "left",
        yanchor = "top"
      )
    )

  marker_params <- list(
    size = ~size,
    showscale = FALSE,
    line = list(width = 3, color = '#FFFFFF')
  )

  for (n in community_levels) {
    community_data <- node_data[node_data$community == n, ]

    plot <- plot %>%
      plotly::add_markers(
        data = community_data,
        x = ~x,
        y = ~y,
        marker = list(
          size = ~size,
          color = palette[n],
          showscale = FALSE,
          line = list(width = 3, color = '#FFFFFF')
        ),
        hoverinfo = 'text',
        text = ~hover_text,
        showlegend = TRUE,
        name = paste("Community", n),
        legendgroup = "Community"
      )
  }

  top_nodes <- head(node_data[order(-node_data$degree), ], top_node_n)

  annotations <- if (nrow(top_nodes) > 0) {
    lapply(1:nrow(top_nodes), function(i) {
      label <- top_nodes$label[i]
      text <- ifelse(!is.na(label) & label != "", label, "")

      list(
        x = top_nodes$x[i],
        y = top_nodes$y[i],
        text = text,
        xanchor = ifelse(!is.na(top_nodes$x[i]) & top_nodes$x[i] > 0, "left", "right"),
        yanchor = ifelse(!is.na(top_nodes$y[i]) & top_nodes$y[i] > 0, "bottom", "top"),
        xshift = ifelse(!is.na(top_nodes$x[i]) & top_nodes$x[i] > 0, 5, -5),
        yshift = ifelse(!is.na(top_nodes$y[i]) & top_nodes$y[i] > 0, 3, -3),
        showarrow = FALSE,
        font = list(size = top_nodes$text_size[i], color = 'black')
      )
    })
  } else {
    list()
  }

  word_co_occurrence_plotly <- plot %>%
    plotly::layout(
      dragmode = "pan",
      title = list(text = "Word Co-occurrence Network", font = list(size = 16)),
      showlegend = TRUE,
      xaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
      yaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
      margin = list(l = 40, r = 100, t = 60, b = 40),
      annotations = annotations
    )

  layout_dff <- layout_df %>%
    dplyr::select(-c("x", "y")) %>%
    dplyr::mutate_if(is.numeric, round, digits = 3)

  summary <- data.frame(
    Metric = c("Nodes", "Edges", "Density", "Diameter",
               "Global Clustering Coefficient", "Local Clustering Coefficient (Mean)",
               "Modularity", "Assortativity", "Geodesic Distance (Mean)"),
    Value = c(
      igraph::vcount(co_occur_graph),
      igraph::ecount(co_occur_graph),
      igraph::graph.density(co_occur_graph),
      igraph::diameter(co_occur_graph),
      igraph::transitivity(co_occur_graph, type = "global"),
      mean(igraph::transitivity(co_occur_graph, type = "local"), na.rm = TRUE),
      igraph::modularity(co_occur_graph, membership = igraph::V(co_occur_graph)$community),
      igraph::assortativity_degree(co_occur_graph),
      mean(igraph::distances(co_occur_graph)[igraph::distances(co_occur_graph) != Inf], na.rm = TRUE)
    )
  ) %>%
    dplyr::mutate_if(is.numeric, round, digits = 3)

  list(
    plot = word_co_occurrence_plotly,
    table = DT::datatable(
      layout_dff,
      rownames = FALSE,
      extensions = 'Buttons',
      options = list(
        scrollX = TRUE,
        scrollY = "400px",
        width = "80%",
        dom = 'Bfrtip',
        buttons = c('copy', 'csv', 'excel', 'pdf', 'print')
      )
    ) %>%
      DT::formatStyle(
        columns = colnames(layout_dff),
        `font-size` = "16px"
      ),
    summary = DT::datatable(summary,
                            rownames = FALSE,
                            extensions = 'Buttons',
                            options = list(
                              scrollX = TRUE,
                              scrollY = "400px",
                              width = "80%",
                              dom = 'Bfrtip',
                              buttons = c('copy', 'csv', 'excel', 'pdf', 'print')
                            )
    ) %>%
      DT::formatStyle(
        columns = colnames(summary),
        `font-size` = "16px"
      )
  )
}

#' @title Analyze and Visualize Word Correlation Networks
#'
#' @description
#' This function creates a word correlation network based on a document-feature matrix (dfm).
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param co_occur_n Minimum number of co-occurrences for filtering terms (default is 30).
#' @param corr_n Minimum correlation value for filtering terms (default is 0.4).
#' @param top_node_n Number of top nodes to display (default is 40).
#' @param height The height of the resulting Plotly plot, in pixels. Defaults to \code{1000}.
#' @param width The width of the resulting Plotly plot, in pixels. Defaults to \code{900}.
#'
#' @return A list containing a Plotly object and a data frame with the results.
#'
#' @importFrom igraph graph_from_data_frame V vcount degree betweenness closeness eigen_centrality layout_with_fr
#' @importFrom plotly plot_ly add_segments add_markers layout
#' @importFrom dplyr count filter mutate select group_by summarise
#' @importFrom tibble as_tibble
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_cor
#' @importFrom scales rescale
#' @importFrom stats quantile
#' @importFrom DT datatable
#' @importFrom shiny showNotification
#' @importFrom utils head
#' @importFrom grDevices colorRampPalette
#' @importFrom RColorBrewer brewer.pal
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'   dfm_object <- quanteda::dfm(tokens)
#'   word_correlation_network_results <- TextAnalysisR::word_correlation_network(
#'                                       dfm_object,
#'                                       co_occur_n = 30,
#'                                       corr_n = 0.4,
#'                                       top_node_n = 40,
#'                                       height = 1000,
#'                                       width = 900)
#'   word_correlation_network_results$plot
#'   word_correlation_network_results$table
#'   word_correlation_network_results$summary
#' }
word_correlation_network <- function(dfm_object,
                                     co_occur_n = 30,
                                     corr_n = 0.4,
                                     top_node_n = 40,
                                     height = 1000,
                                     width = 900) {

  dfm_td <- tidytext::tidy(dfm_object) %>%
    tibble::as_tibble()

  term_cor <- dfm_td %>%
    tibble::as_tibble() %>%
    group_by(term) %>%
    filter(n() >= co_occur_n) %>%
    widyr::pairwise_cor(term, document, sort = TRUE)

  term_cor_graph <- term_cor %>%
    filter(correlation > corr_n) %>%
    igraph::graph_from_data_frame(directed = FALSE)

  if (igraph::vcount(term_cor_graph) == 0) {
    showNotification("No correlation relationships meet the threshold.", type = "error")
    return(NULL)
  }

  igraph::V(term_cor_graph)$degree <- igraph::degree(term_cor_graph)
  igraph::V(term_cor_graph)$betweenness <- igraph::betweenness(term_cor_graph)
  igraph::V(term_cor_graph)$closeness <- igraph::closeness(term_cor_graph)
  igraph::V(term_cor_graph)$eigenvector <- igraph::eigen_centrality(term_cor_graph)$vector
  igraph::V(term_cor_graph)$community <- igraph::cluster_leiden(term_cor_graph)$membership

  layout <- igraph::layout_with_fr(term_cor_graph)
  layout_df <- as.data.frame(layout)
  colnames(layout_df) <- c("x", "y")

  layout_df$label <- igraph::V(term_cor_graph)$name
  layout_df$degree <- igraph::V(term_cor_graph)$degree
  layout_df$betweenness <- igraph::V(term_cor_graph)$betweenness
  layout_df$closeness <- igraph::V(term_cor_graph)$closeness
  layout_df$eigenvector <- igraph::V(term_cor_graph)$eigenvector
  layout_df$community <- igraph::V(term_cor_graph)$community

  edge_data <- igraph::as_data_frame(term_cor_graph, what = "edges") %>%
    dplyr::mutate(
      x = layout_df$x[match(from, layout_df$label)],
      y = layout_df$y[match(from, layout_df$label)],
      xend = layout_df$x[match(to, layout_df$label)],
      yend = layout_df$y[match(to, layout_df$label)],
      correlation = correlation
    ) %>%
    dplyr::select(from, to, x, y, xend, yend, correlation)

  edge_data <- edge_data %>%
    dplyr::mutate(
      line_group = as.integer(cut(
        correlation,
        breaks = unique(quantile(correlation, probs = seq(0, 1, length.out = 6), na.rm = TRUE)),
        include.lowest = TRUE
      )),
      line_width = scales::rescale(line_group, to = c(1, 5)),
      alpha = scales::rescale(line_group, to = c(0.1, 0.3))
    )

  edge_group_labels <- edge_data %>%
    dplyr::group_by(line_group) %>%
    dplyr::summarise(
      min_corr = min(correlation, na.rm = TRUE),
      max_corr = max(correlation, na.rm = TRUE)
    ) %>%
    dplyr::mutate(label = paste0("Correlation: ", round(min_corr, 2), " - ", round(max_corr, 2))) %>%
    dplyr::pull(label)

  node_data <- layout_df %>%
    dplyr::mutate(
      degree_log = log1p(degree),
      size = scales::rescale(degree_log, to = c(12, 30)),
      text_size = scales::rescale(degree_log, to = c(14, 20)),
      alpha = scales::rescale(degree_log, to = c(0.2, 1)),
      hover_text = paste(
        "Word:", label,
        "<br>Degree:", degree,
        "<br>Betweenness:", round(betweenness, 2),
        "<br>Closeness:", round(closeness, 2),
        "<br>Eigenvector:", round(eigenvector, 2),
        "<br>Community:", community
      )
    )

  n_communities <- length(unique(node_data$community))
  if (n_communities >= 3 && n_communities <= 8) {
    palette <- RColorBrewer::brewer.pal(n_communities, "Set2")
  } else if (n_communities > 8) {
    palette <- colorRampPalette(RColorBrewer::brewer.pal(8, "Set2"))(n_communities)
  } else if (n_communities > 0 && n_communities < 3) {
    palette <- RColorBrewer::brewer.pal(3, "Set2")[1:n_communities]
  } else {
    palette <- rep("#000000", n_communities)
  }

  node_data$community <- factor(node_data$community, levels = unique(node_data$community))
  community_levels <- levels(node_data$community)
  names(palette) <- community_levels
  node_data$color <- palette[as.character(node_data$community)]

  plot <- plotly::plot_ly(
    type = 'scatter',
    mode = 'markers',
    width = width,
    height = height
  )

  for (i in unique(edge_data$line_group)) {

    edge_subset <- edge_data %>% dplyr::filter(line_group == i)
    edge_label <- edge_group_labels[i]

    edge_subset <- edge_subset %>%
      dplyr::mutate(
        mid_x = (x + xend) / 2,
        mid_y = (y + yend) / 2
      )

    if (nrow(edge_subset) > 0) {
      plot <- plot %>%
        plotly::add_segments(
          data = edge_subset,
          x = ~x,
          y = ~y,
          xend = ~xend,
          yend = ~yend,
          line = list(
            color = '#5C5CFF',
            width = ~line_width
          ),
          hoverinfo = 'none',
          opacity = ~alpha,
          showlegend = TRUE,
          name = edge_label,
          legendgroup = "Edges"
        ) %>%

        plotly::add_trace(
          data = edge_subset,
          x = ~mid_x,
          y = ~mid_y,
          type = 'scatter',
          mode = 'markers',
          marker = list(size = 0.1, color = '#e0f7ff', opacity = 0),
          text = ~paste0(
            "Correlation:", round(correlation, 2),
            "<br>Source: ", from,
            "<br>Target: ", to
          ),
          hoverinfo = 'text',
          showlegend = FALSE
        )
    }
  }

  plot <- plot %>%
    plotly::layout(
      legend = list(
        title = list(text = "Correlation"),
        orientation = "v",
        x = 1.1,
        y = 1,
        xanchor = "left",
        yanchor = "top"
      )
    )

  marker_params <- list(
    size = ~size,
    showscale = FALSE,
    line = list(width = 3, color = '#FFFFFF')
  )

  for (n in community_levels) {
    community_data <- node_data[node_data$community == n, ]

    plot <- plot %>%
      plotly::add_markers(
        data = community_data,
        x = ~x,
        y = ~y,
        marker = list(
          size = ~size,
          color = palette[n],
          showscale = FALSE,
          line = list(width = 3, color = '#FFFFFF')
        ),
        hoverinfo = 'text',
        text = ~hover_text,
        showlegend = TRUE,
        name = paste("Community", n),
        legendgroup = "Community"
      )
  }

  top_nodes <- head(node_data[order(-node_data$degree), ], top_node_n)

  annotations <- if (nrow(top_nodes) > 0) {
    lapply(1:nrow(top_nodes), function(i) {
      label <- top_nodes$label[i]
      text <- ifelse(!is.na(label) & label != "", label, "")

      list(
        x = top_nodes$x[i],
        y = top_nodes$y[i],
        text = text,
        xanchor = ifelse(!is.na(top_nodes$x[i]) & top_nodes$x[i] > 0, "left", "right"),
        yanchor = ifelse(!is.na(top_nodes$y[i]) & top_nodes$y[i] > 0, "bottom", "top"),
        xshift = ifelse(!is.na(top_nodes$x[i]) & top_nodes$x[i] > 0, 5, -5),
        yshift = ifelse(!is.na(top_nodes$y[i]) & top_nodes$y[i] > 0, 3, -3),
        showarrow = FALSE,
        font = list(size = top_nodes$text_size[i], color = 'black')
      )
    })
  } else {
    list()
  }

  word_correlation_plotly <- plot %>%
    plotly::layout(
      dragmode = "pan",
      title = list(text = "Word Correlation Network", font = list(size = 16)),
      showlegend = TRUE,
      xaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
      yaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
      margin = list(l = 40, r = 100, t = 60, b = 40),
      annotations = annotations
    )

  layout_dff <- layout_df %>%
    dplyr::select(-c("x", "y")) %>%
    dplyr::mutate_if(is.numeric, round, digits = 3)

  summary <- data.frame(
    Metric = c("Nodes", "Edges", "Density", "Diameter",
               "Global Clustering Coefficient", "Local Clustering Coefficient (Mean)",
               "Modularity", "Assortativity", "Geodesic Distance (Mean)"),
    Value = c(
      igraph::vcount(term_cor_graph),
      igraph::ecount(term_cor_graph),
      igraph::graph.density(term_cor_graph),
      igraph::diameter(term_cor_graph),
      igraph::transitivity(term_cor_graph, type = "global"),
      mean(igraph::transitivity(term_cor_graph, type = "local"), na.rm = TRUE),
      igraph::modularity(term_cor_graph, membership = igraph::V(term_cor_graph)$community),
      igraph::assortativity_degree(term_cor_graph),
      mean(igraph::distances(term_cor_graph)[igraph::distances(term_cor_graph) != Inf], na.rm = TRUE)
    )
  ) %>%
    dplyr::mutate_if(is.numeric, round, digits = 3)

  list(
    plot = word_correlation_plotly,
    table = DT::datatable(
      layout_dff,
      rownames = FALSE,
      extensions = 'Buttons',
      options = list(
        scrollX = TRUE,
        scrollY = "400px",
        width = "80%",
        dom = 'Bfrtip',
        buttons = c('copy', 'csv', 'excel', 'pdf', 'print')
      )
    ) %>%
      DT::formatStyle(
        columns = colnames(layout_dff),
        `font-size` = "16px"
      ),
    summary = DT::datatable(summary,
                            rownames = FALSE,
                            extensions = 'Buttons',
                            options = list(
                              scrollX = TRUE,
                              scrollY = "400px",
                              width = "80%",
                              dom = 'Bfrtip',
                              buttons = c('copy', 'csv', 'excel', 'pdf', 'print')
                            )
    ) %>%
      DT::formatStyle(
        columns = colnames(summary),
        `font-size` = "16px"
      )
  )
}


#' @title Analyze and Visualize Term Proportions by a Continuous Variable
#'
#' @description
#' This function analyzes and visualizes term proportions by a continuous variable.
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param stm_model An STM model object.
#' @param continuous_variable A continuous variable in the metadata.
#' @param selected_terms A vector of terms to analyze trends for.
#' @param height The height of the resulting Plotly plot, in pixels. Defaults to \code{500}.
#' @param width The width of the resulting Plotly plot, in pixels. Defaults to \code{900}.
#'
#' @return A list containing Plotly objects and tables with the results.
#'
#' @details This function requires a fitted STM model object and a quanteda dfm object.
#'
#' @importFrom stats glm reformulate binomial
#' @importFrom plotly ggplotly
#' @importFrom DT datatable
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'   dfm_object <- quanteda::dfm(tokens)
#'   stm_15 <- TextAnalysisR::stm_15
#'   term_proportion_results <- TextAnalysisR::term_proportion(
#'                              dfm_object,
#'                              stm_model = stm_15,
#'                              continuous_variable = "year",
#'                              selected_terms = c("calculator", "computer"),
#'                              height = 500,
#'                              width = 900)
#'   term_proportion_results$plot
#'   term_proportion_results$table
#' }
term_proportion <- function(dfm_object,
                            stm_model,
                            continuous_variable,
                            selected_terms,
                            height = 500,
                            width = 900) {

  if (requireNamespace("htmltools", quietly = TRUE)) {

    dfm_outcome_obj <- dfm_object
    dfm_td <- tidytext::tidy(dfm_object)
    gamma_td <-
      tidytext::tidy(
        stm_model,
        matrix = "gamma",
        document_names = rownames(dfm_object)
      )
    dfm_outcome_obj@docvars$document <- dfm_outcome_obj@docvars$docname_

    dfm_gamma_td <- gamma_td %>%
      left_join(dfm_outcome_obj@docvars,
                by = c("document" = "document")) %>%
      left_join(dfm_td, by = c("document" = "document"), relationship = "many-to-many")

    con_var_term_counts <- dfm_gamma_td %>%
      tibble::as_tibble() %>%
      group_by(!!rlang::sym(continuous_variable)) %>%
      mutate(
        con_3_total = sum(count),
        term_proportion = count / con_3_total
      ) %>%
      ungroup()

    con_var_term_gg <- con_var_term_counts %>%
      mutate(term = factor(term, levels = selected_terms)) %>%
      mutate(across(where(is.numeric), ~ round(., 3))) %>%
      filter(term %in% selected_terms) %>%
      ggplot(aes(
        x = !!rlang::sym(continuous_variable),
        y = term_proportion,
        group = term
      )) +
      geom_point(color = "#337ab7", alpha = 0.6, size = 1) +
      geom_line(color = "#337ab7", alpha = 0.6, linewidth = 0.5) +
      facet_wrap(~ term, scales = "free") +
      scale_y_continuous(labels = scales::percent_format()) +
      labs(y = "Term Proportion (%)") +
      theme_minimal(base_size = 11) +
      theme(
        legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
        axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
        strip.text.x = element_text(size = 11, color = "#3B3B3B"),
        axis.text.x = element_text(size = 11, color = "#3B3B3B"),
        axis.text.y = element_text(size = 11, color = "#3B3B3B"),
        axis.title = element_text(size = 11, color = "#3B3B3B"),
        axis.title.x = element_text(margin = margin(t = 9)),
        axis.title.y = element_text(margin = margin(r = 11))
      )

    con_var_term_plotly <- plotly::ggplotly(
      con_var_term_gg,
      height = height,
      width = width
    ) %>%
      plotly::layout(
        margin = list(l = 40, r = 150, t = 60, b = 40)
      )

    significance_results <- con_var_term_counts %>%
      mutate(word = term) %>%
      filter(word %in% selected_terms) %>%
      group_by(word) %>%
      do(
        tidy(
          glm(
            cbind(count, con_3_total - count) ~ s(!!rlang::sym(continuous_variable)),
            weights = con_3_total,
            family = binomial(link = "logit"),
            data = .
          )
        )
      ) %>%
      mutate(`odds ratio` = exp(estimate)) %>%
      rename(`logit` = estimate) %>%
      ungroup()

    significance_results_tables <- significance_results %>%
      mutate(word = factor(word, levels = selected_terms)) %>%
      arrange(word) %>%
      group_by(word) %>%
      group_map(~ {
        htmltools::tagList(
          htmltools::tags$div(
            style = "margin-bottom: 20px;",
            htmltools::tags$p(
              style = "font-weight: bold; text-align: center;",
              .y$word
            )
          ),
          .x %>%
            mutate_if(is.numeric, ~ round(., 3)) %>%
            DT::datatable(
              rownames = FALSE,
              extensions = 'Buttons',
              options = list(
                scrollX = TRUE,
                scrollY = "400px",
                width = "80%",
                dom = 'Bfrtip',
                buttons = c('copy', 'csv', 'excel', 'pdf', 'print')
              )
            )
        )
      })

    list(
      plot = con_var_term_plotly,
      table = htmltools::tagList(significance_results_tables) %>% htmltools::browsable()
    )

  } else {
    stop("htmltools is required for rendering this report. Please install it.")
  }

}
