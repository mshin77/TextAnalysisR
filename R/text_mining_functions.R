
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
#'
#' @importFrom quanteda convert
#' @importFrom stm searchK
#' @importFrom plotly plot_ly subplot layout
#' @importFrom dplyr mutate select
#' @importFrom stats as.formula
#' @importFrom utils str
evaluate_optimal_topic_number <- function(dfm_object,
                                          topic_range,
                                          max.em.its = 75,
                                          categorical_var = NULL,
                                          continuous_var = NULL,
                                          height = 600,
                                          width = 800,
                                          verbose = TRUE, ...) {

  if (!is.null(categorical_var) && !(categorical_var %in% names(quanteda::docvars(dfm_object)))) {
    stop(paste("Categorical variable", categorical_var, "is missing in dfm_object's document variables."))
  }

  if (!is.null(continuous_var) && !(continuous_var %in% names(quanteda::docvars(dfm_object)))) {
    stop(paste("Continuous variable", continuous_var, "is missing in dfm_object's document variables."))
  }

  out <- quanteda::convert(dfm_object, to = "stm")

  if (!all(c("meta", "documents", "vocab") %in% names(out))) {
    stop("Conversion of dfm_object must result in 'meta', 'documents', and 'vocab'.")
  }

  meta <- out$meta
  documents <- out$documents
  vocab <- out$vocab

  if (nrow(meta) != length(documents)) {
    stop("Number of rows in 'meta' does not match number of 'documents' after conversion.")
  }

  if (!is.null(categorical_var)) {
    meta[[categorical_var]] <- as.factor(meta[[categorical_var]])
  }
  if (!is.null(continuous_var)) {
    meta[[continuous_var]] <- as.numeric(meta[[continuous_var]])
    if (any(is.na(meta[[continuous_var]]))) {
      stop(paste("Continuous variable", continuous_var, "contains NA values. Please handle them before proceeding."))
    }
  }

  message("Variable types in metadata:")
  # print(str(meta))

  terms <- c()
  if (!is.null(categorical_var)) {
    terms <- c(terms, categorical_var)
  }
  if (!is.null(continuous_var)) {
    terms <- c(terms, paste0("s(", continuous_var, ")"))
  }
  if (length(terms) > 0) {
    formula_string <- paste("~", paste(terms, collapse = " + "))
    prevalence_formula <- as.formula(formula_string)
    message("Prevalence formula: ", deparse(prevalence_formula))
  } else {
    prevalence_formula <- NULL
    message("No prevalence formula specified.")
  }

  search_result <- tryCatch({
    stm::searchK(
      data = meta,
      documents = documents,
      vocab = vocab,
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
    hoverinfo = 'text'
  )

  p2 <- plotly::plot_ly(
    data = search_result$results,
    x = ~K,
    y = ~residual,
    type = 'scatter',
    mode = 'lines+markers',
    text = ~paste("K:", K, "<br>Residuals:", round(residual, 3)),
    hoverinfo = 'text'
  )

  p3 <- plotly::plot_ly(
    data = search_result$results,
    x = ~K,
    y = ~semcoh,
    type = 'scatter',
    mode = 'lines+markers',
    text = ~paste("K:", K, "<br>Semantic Coherence:", round(semcoh, 3)),
    hoverinfo = 'text'
  )

  p4 <- plotly::plot_ly(
    data = search_result$results,
    x = ~K,
    y = ~lbound,
    type = 'scatter',
    mode = 'lines+markers',
    text = ~paste("K:", K, "<br>Lower Bound:", round(lbound, 3)),
    hoverinfo = 'text'
  )

  plotly::subplot(p1, p2, p3, p4, nrows = 2, margin = 0.1) %>%
    plotly::layout(
      height = height,
      width = width,
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
#'
#' @importFrom stats reorder
#' @importFrom numform ff_num
#' @importFrom plotly ggplotly layout
#' @importFrom tidytext reorder_within scale_x_reordered
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
    geom_col(show.legend = FALSE, alpha = 0.8) +
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
#'   width = 1000,
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
                                       width = 1000,
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
    mutate(topic = as.integer(as.character(topic))) %>%  # Ensure topic is integer
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
    geom_col(alpha = 0.8) +
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


#' @title Plot a Word Co-occurrence Network
#'
#' @description
#' Visualize the co-occurrence relationships between terms in the corpus based on pairwise counts.
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param co_occur_n Minimum number of co-occurrences for filtering terms (default is 200).
#' @param height The height of the resulting Plotly plot, in pixels. Defaults to \code{900}.
#' @param width The width of the resulting Plotly plot, in pixels. Defaults to \code{800}.
#'
#' @return A Plotly object visualizing the interactive word co-occurrence network.
#' @export
#'
#' @importFrom igraph graph_from_data_frame V vcount degree betweenness closeness eigen_centrality layout_with_fr
#' @importFrom plotly plot_ly add_segments add_markers layout
#' @importFrom dplyr count filter mutate select group_by summarise
#' @importFrom tibble as_tibble
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_count
#' @importFrom scales rescale
#' @importFrom stats quantile
#' @importFrom shiny showNotification
#' @importFrom rlang sym
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'   dfm_object <- quanteda::dfm(tokens)
#'   TextAnalysisR::plot_word_co_occurrence_network(
#'     dfm_object,
#'     co_occur_n = 200,
#'     height = 900,
#'     width = 800)
#' }
plot_word_co_occurrence_network <- function(dfm_object,
                                            co_occur_n = 200,
                                            height = 900,
                                            width = 800) {

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

  layout <- igraph::layout_with_fr(co_occur_graph)
  layout_df <- as.data.frame(layout)
  colnames(layout_df) <- c("x", "y")
  layout_df$label <- igraph::V(co_occur_graph)$name
  layout_df$degree <- igraph::V(co_occur_graph)$degree
  layout_df$betweenness <- igraph::V(co_occur_graph)$betweenness
  layout_df$closeness <- igraph::V(co_occur_graph)$closeness
  layout_df$eigenvector <- igraph::V(co_occur_graph)$eigenvector

  edge_data <- igraph::as_data_frame(co_occur_graph, what = "edges") %>%
    mutate(
      x = layout_df$x[match(from, layout_df$label)],
      y = layout_df$y[match(from, layout_df$label)],
      xend = layout_df$x[match(to, layout_df$label)],
      yend = layout_df$y[match(to, layout_df$label)],
      cooccur_count = n
    ) %>%
    select(x, y, xend, yend, cooccur_count)

  edge_data <- edge_data %>%
    mutate(
      line_group = as.integer(cut(
        cooccur_count,
        breaks = unique(quantile(cooccur_count, probs = seq(0, 1, length.out = 6), na.rm = TRUE)),
        include.lowest = TRUE
      )),
      line_width = scales::rescale(line_group, to = c(2, 10)),
      alpha = scales::rescale(line_group, to = c(0.2, 0.6))
    )

  edge_group_labels <- edge_data %>%
    group_by(line_group) %>%
    summarise(
      min_count = min(cooccur_count, na.rm = TRUE),
      max_count = max(cooccur_count, na.rm = TRUE)
    ) %>%
    mutate(label = paste0("Count: ", min_count, " - ", max_count)) %>%
    pull(label)

  node_data <- layout_df %>%
    mutate(
      degree_log = log1p(degree),
      size = scales::rescale(degree_log, to = c(12, 30)),
      text_size = scales::rescale(degree_log, to = c(14, 20)),
      alpha = scales::rescale(degree_log, to = c(0.2, 0.9)),
      hover_text = paste(
        "Word:", label,
        "<br>Degree:", degree,
        "<br>Betweenness:", round(betweenness, 2),
        "<br>Closeness:", round(closeness, 2),
        "<br>Eigenvector:", round(eigenvector, 2)
      )
    )

  color_scale <- 'Viridis'

  plot <- plotly::plot_ly(
    type = 'scatter',
    mode = 'markers',
    width = width,
    height = height
  )

  for (i in unique(edge_data$line_group)) {
    edge_subset <- edge_data %>% filter(line_group == i)
    edge_label <- edge_group_labels[i]

    if (nrow(edge_subset) > 0) {
      plot <- plot %>%
        plotly::add_segments(
          data = edge_subset,
          x = ~x,
          y = ~y,
          xend = ~xend,
          yend = ~yend,
          line = list(color = 'rgba(0, 0, 255, 0.6)', width = ~line_width),
          hoverinfo = 'text',
          text = ~paste("Co-occurrence:", cooccur_count),
          opacity = ~alpha,
          showlegend = TRUE,
          name = edge_label,
          legendgroup = "Edges"
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
    color = ~degree,
    colorscale = color_scale,
    showscale = TRUE,
    colorbar = list(
      title = "Degree Centrality",
      len = 0.25,
      x = 1.1,
      y = 0.6
    ),
    line = list(width = 2, color = '#FFFFFF'),
    opacity = ~alpha
  )

  plot <- plot %>%
    plotly::add_markers(
      data = node_data,
      x = ~x,
      y = ~y,
      marker = marker_params,
      hoverinfo = 'text',
      text = ~hover_text,
      showlegend = FALSE
    )

  annotations <- lapply(1:nrow(node_data), function(i) {
    list(
      x = node_data$x[i],
      y = node_data$y[i],
      text = node_data$label[i],
      xanchor = "center",
      yanchor = "bottom",
      showarrow = FALSE,
      font = list(size = node_data$text_size[i], color = 'black')
    )
  })

  plot <- plot %>%
    plotly::layout(
      dragmode = "pan",
      title = list(text = "Word Co-occurrence Network", font = list(size = 16)),
      showlegend = TRUE,
      xaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
      yaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
      margin = list(l = 40, r = 100, t = 60, b = 40),
      annotations = annotations
    )

  plot
}


#' @title Plot a Word Correlation Network
#'
#' @description
#' Visualize the correlation relationships between terms in the corpus based on pairwise correlations.
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param co_occur_n Minimum number of co-occurrences for filtering terms (default is 30).
#' @param corr_n Minimum correlation value for filtering terms (default is 0.4).
#' @param height The height of the resulting Plotly plot, in pixels. Defaults to \code{900}.
#' @param width The width of the resulting Plotly plot, in pixels. Defaults to \code{800}.
#'
#' @return A Plotly object visualizing the interactive word correlation network.
#' @export
#'
#' @importFrom igraph graph_from_data_frame V vcount degree betweenness closeness eigen_centrality layout_with_fr
#' @importFrom plotly plot_ly add_segments add_markers layout
#' @importFrom dplyr count filter mutate select group_by summarise
#' @importFrom tibble as_tibble
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_cor
#' @importFrom scales rescale
#' @importFrom stats quantile
#' @importFrom shiny showNotification
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'   dfm_object <- quanteda::dfm(tokens)
#'   TextAnalysisR::plot_word_correlation_network(
#'     dfm_object,
#'     co_occur_n = 30,
#'     corr_n = 0.4,
#'     height = 900,
#'     width = 800)
#' }
plot_word_correlation_network <- function(dfm_object,
                                          co_occur_n = 30,
                                          corr_n = 0.4,
                                          height = 900,
                                          width = 800) {

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

  layout <- igraph::layout_with_fr(term_cor_graph)
  layout_df <- as.data.frame(layout)
  colnames(layout_df) <- c("x", "y")
  layout_df$label <- igraph::V(term_cor_graph)$name
  layout_df$degree <- igraph::V(term_cor_graph)$degree
  layout_df$betweenness <- igraph::V(term_cor_graph)$betweenness
  layout_df$closeness <- igraph::V(term_cor_graph)$closeness
  layout_df$eigenvector <- igraph::V(term_cor_graph)$eigenvector

  edge_data <- igraph::as_data_frame(term_cor_graph, what = "edges") %>%
    mutate(
      x = layout_df$x[match(from, layout_df$label)],
      y = layout_df$y[match(from, layout_df$label)],
      xend = layout_df$x[match(to, layout_df$label)],
      yend = layout_df$y[match(to, layout_df$label)],
      correlation = correlation
    ) %>%
    select(x, y, xend, yend, correlation)

  edge_data <- edge_data %>%
    mutate(
      line_group = as.integer(cut(
        correlation,
        breaks = unique(quantile(correlation, probs = seq(0, 1, length.out = 6), na.rm = TRUE)),
        include.lowest = TRUE
      )),
      line_width = scales::rescale(line_group, to = c(2, 10)),
      alpha = scales::rescale(line_group, to = c(0.2, 0.6))
    )

  edge_group_labels <- edge_data %>%
    group_by(line_group) %>%
    summarise(
      min_corr = min(correlation, na.rm = TRUE),
      max_corr = max(correlation, na.rm = TRUE)
    ) %>%
    mutate(label = paste0("Correlation: ", round(min_corr, 2), " - ", round(max_corr, 2))) %>%
    pull(label) %>%
    unname()

  node_data <- layout_df %>%
    mutate(
      degree_log = log1p(degree),
      size = scales::rescale(degree_log, to = c(12, 30)),
      text_size = scales::rescale(degree_log, to = c(14, 20)),
      alpha = scales::rescale(degree_log, to = c(0.2, 0.9)),
      hover_text = paste(
        "Word:", label,
        "<br>Degree:", degree,
        "<br>Betweenness:", round(betweenness, 2),
        "<br>Closeness:", round(closeness, 2),
        "<br>Eigenvector:", round(eigenvector, 2)
      )
    )

  color_scale <- 'Viridis'

  plot <- plotly::plot_ly(
    type = 'scatter',
    mode = 'markers',
    width = width,
    height = height
  )

  for (i in unique(edge_data$line_group)) {
    edge_subset <- edge_data %>% filter(line_group == i)
    edge_label <- edge_group_labels[i]

    if (nrow(edge_subset) > 0) {
      plot <- plot %>%
        add_segments(
          data = edge_subset,
          x = ~x,
          y = ~y,
          xend = ~xend,
          yend = ~yend,
          line = list(color = 'rgba(0, 0, 255, 0.6)', width = ~line_width),
          hoverinfo = 'text',
          text = ~paste("Correlation:", round(correlation, 2)),
          opacity = ~alpha,
          showlegend = TRUE,
          name = edge_label,
          legendgroup = "Edges"
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
    color = ~degree,
    colorscale = color_scale,
    showscale = TRUE,
    colorbar = list(
      title = "Degree Centrality",
      len = 0.25,
      x = 1.1,
      y = 0.6
    ),
    line = list(width = 2, color = '#FFFFFF'),
    opacity = ~alpha
  )

  plot <- plot %>%
    plotly::add_markers(
      data = node_data,
      x = ~x,
      y = ~y,
      marker = marker_params,
      hoverinfo = 'text',
      text = ~hover_text,
      showlegend = FALSE
    )

  annotations <- lapply(1:nrow(node_data), function(i) {
    list(
      x = node_data$x[i],
      y = node_data$y[i],
      text = node_data$label[i],
      xanchor = "center",
      yanchor = "bottom",
      showarrow = FALSE,
      font = list(size = node_data$text_size[i], color = 'black')
    )
  })

  plot <- plot %>%
    plotly::layout(
      dragmode = "pan",
      title = list(text = "Word Correlation Network", font = list(size = 16)),
      showlegend = TRUE,
      xaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
      yaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
      margin = list(l = 40, r = 150, t = 60, b = 40),
      annotations = annotations
    )

  plot
}


#' @title Plot Word Frequency Trends Over Time
#'
#' @description Analyze and visualize word frequency trends over time for a fixed term column.
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param stm_model An STM model object.
#' @param time_variable The column name for the time variable (e.g., "year").
#' @param selected_terms A vector of terms to analyze trends for.
#' @param height The height of the resulting Plotly plot, in pixels. Defaults to \code{500}.
#' @param width The width of the resulting Plotly plot, in pixels. Defaults to \code{1000}.
#'
#' @return A Plotly object showing interactive word frequency trends over time.
#'
#' @details This function requires a fitted STM model object and a quanteda dfm object.
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
#'   TextAnalysisR::word_frequency_trends(dfm_object,
#'                                     stm_model = stm_15,
#'                                     time_variable = "year",
#'                                     selected_terms = c("calculator", "computer"),
#'                                     height = 500,
#'                                     width = 1000)
#' }
#'
#' @importFrom stats glm reformulate
#' @importFrom plotly ggplotly
word_frequency_trends <- function(dfm_object,
                                  stm_model,
                                  time_variable,
                                  selected_terms,
                                  height = 500,
                                  width = 1000) {
  dfm_td <- tidytext::tidy(dfm_object) %>%
    tibble::as_tibble()

  gamma_td <- tidytext::tidy(stm_model, matrix = "gamma", document_names = quanteda::docnames(dfm_object)) %>%
    tibble::as_tibble()

  if (any(is.na(gamma_td$document))) {
    stop("Some document IDs in the STM model do not have corresponding names in dfm_object.")
  }

  dfm_object@docvars$document <- dfm_object@docvars$docname_

  docvars_df <- quanteda::docvars(dfm_object) %>%
    dplyr::select(document, everything()) %>%
    dplyr::distinct(document, .keep_all = TRUE)

  dfm_gamma_td <- gamma_td %>%
    dplyr::left_join(docvars_df, by = "document") %>%
    dplyr::left_join(dfm_td, by = "document", relationship = "many-to-many")

  if (!"document" %in% names(dfm_gamma_td)) {
    stop("'document' column is missing after joins.")
  }

  year_term_counts <- dfm_gamma_td %>%
    tibble::as_tibble() %>%
    dplyr::group_by(!!rlang::sym(time_variable)) %>%
    dplyr::mutate(
      total_count = sum(count),
      term_proportion = count / total_count
    ) %>%
    dplyr::ungroup()

  year_term_gg <- year_term_counts %>%
    dplyr::mutate(across(where(is.numeric), ~ round(., 3))) %>%
    dplyr::filter(term %in% selected_terms) %>%
    ggplot2::ggplot(ggplot2::aes(
      x = !!rlang::sym(time_variable),
      y = term_proportion,
      group = term,
      text = paste0("Term Proportion: ", sprintf("%.3f", term_proportion))
    )) +
    ggplot2::geom_point(color = "#636363", alpha = 0.6, size = 1) +
    ggplot2::geom_smooth(color = "#337ab7",
                         se = TRUE,
                         method = "loess",
                         linewidth = 0.5,
                         formula = y ~ x) +
    ggplot2::facet_wrap(~ term, scales = "free_y") +
    ggplot2::scale_y_continuous(labels = scales::percent_format()) +
    ggplot2::labs(x = "", y = "") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      legend.position = "none",
      panel.grid.major = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      axis.line = ggplot2::element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.ticks = ggplot2::element_line(color = "#3B3B3B", linewidth = 0.3),
      strip.text.x = ggplot2::element_text(size = 11, color = "#3B3B3B"),
      axis.text.x = ggplot2::element_text(size = 11, color = "#3B3B3B"),
      axis.text.y = ggplot2::element_text(size = 11, color = "#3B3B3B"),
      axis.title = ggplot2::element_text(size = 11, color = "#3B3B3B"),
      axis.title.x = ggplot2::element_text(margin = ggplot2::margin(t = 9)),
      axis.title.y = ggplot2::element_text(margin = ggplot2::margin(r = 9))
    )

  plotly::ggplotly(
    year_term_gg,
    height = height,
    width = width,
    tooltip = "text"
  ) %>%
    plotly::layout(
      margin = list(l = 40, r = 150, t = 60, b = 40)
    )
}







