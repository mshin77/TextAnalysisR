################################################################################
# PIPE OPERATOR
################################################################################

#' Pipe operator
#'
#' See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @usage lhs \%>\% rhs
#' @param lhs A value or the magrittr placeholder.
#' @param rhs A function call using the magrittr semantics.
#' @return The result of calling `rhs(lhs)`.
NULL



################################################################################
# GLOBAL VARIABLES (for R CMD check)
################################################################################

globalVariables(names = c(
  ".", ".max_idx", ".row_idx", "categorical_var", "centrality", "col_idx",
  "collocation", "community", "continuous_var", "cooccur_count", "correlation",
  "correlation_rounded", "degree_log", "document", "eigenvector", "emotion",
  "entity", "estimate", "feature", "frequency", "from", "generated_content", "group", "size_metric_log",
  "interview_question", "item1", "item2", "label", "labeled_topic", "line_group",
  "lower", "max_corr", "max_count", "max_similarity", "metric_value", "min_corr",
  "min_count", "model_type", "n", "n_words", "name", "negative", "odds ratio",
  "odds.ratio", "ord", "other_category", "other_id", "other_idx", "other_name",
  "p.value", "percent", "policy_recommendation", "pos", "positive", "proportion",
  "ref_id", "ref_idx", "ref_name", "research_question", "row_idx", "score",
  "sentiment", "sentiment_score", "similarity", "statistic", "std.error",
  "std.error (odds ratio)", "stopwords", "survey_item", "term", "term_proportion",
  "terms", "text", "theme_description", "to", "tokens_remove", "tokens_select",
  "topic", "topic_display", "topic_label", "total_count", "total_score", "tt",
  "united_texts", "upper", "value", "word", "word_frequency", "x", "xend",
  "y", "yend"
))



#' @importFrom utils modifyList
#' @importFrom stats cor
NULL

# Utility and Helper Functions
# General-purpose utility functions for analysis and visualization

#
# Deployment Detection Utilities
#

#' Check Docker Deployment
#'
#' @description
#' Detects whether the app is running in a Docker container.
#' Docker deployments have full Python/spaCy capability.
#'
#' @return Logical TRUE if running in Docker
#'
#' @keywords internal
check_docker_deployment <- function() {
  has_dockerenv <- file.exists("/.dockerenv")
  has_docker_env_var <- nzchar(Sys.getenv("TEXTANALYSISR_DOCKER"))
  has_docker_container <- nzchar(Sys.getenv("DOCKER_CONTAINER"))

  return(has_dockerenv || has_docker_env_var || has_docker_container)
}

#' Check Deployment Environment
#'
#' @description
#' Detects whether the app is running on a web server (shinyapps.io, Posit Connect)
#' versus locally via `run_app()`.
#'
#' @return Logical TRUE if running on web server, FALSE if local
#'
#' @export
#'
#' @examples
#' if (check_web_deployment()) {
#'   message("Running on web - some features disabled")
#' }
check_web_deployment <- function() {
  # Docker has Python/spaCy available - not a restricted deployment
  if (check_docker_deployment()) {
    return(FALSE)
  }

  # Check for restricted web servers (no Python)
  shinyapps <- Sys.getenv("R_CONFIG_ACTIVE") == "shinyapps"
  shinyapps_io <- grepl("shinyapps", Sys.getenv("SHINY_SERVER_URL", ""), ignore.case = TRUE)
  connect <- nzchar(Sys.getenv("RSTUDIO_CONNECT_HASTE"))

  return(shinyapps || shinyapps_io || connect)
}

#' Check Feature Status
#'
#' @description
#' Checks if a specific optional feature is available in the current environment.
#'
#' @param feature Character: "python", "ollama", "pdf_tables", "embeddings", "sentiment_deep"
#'
#' @return Logical TRUE if feature is available
#'
#' @export
#'
#' @examples
#' if (check_feature("ollama")) {
#'   # Use AI-powered labeling
#' }
check_feature <- function(feature) {
  feature <- tolower(feature)

  if (check_web_deployment()) {
    return(feature %in% c("core", "lexical", "stm"))
  }

  switch(feature,
    "python" = tryCatch({
      status <- check_python_env()
      isTRUE(status$available)
    }, error = function(e) FALSE),
    "ollama" = tryCatch({
      check_ollama(verbose = FALSE)
    }, error = function(e) FALSE),
    "pdf_tables" = tryCatch({
      status <- check_python_env()
      isTRUE(status$available) && isTRUE(status$packages$pdfplumber)
    }, error = function(e) FALSE),
    "embeddings" = tryCatch({
      requireNamespace("reticulate", quietly = TRUE) &&
        reticulate::py_module_available("sentence_transformers")
    }, error = function(e) FALSE),
    "sentiment_deep" = tryCatch({
      requireNamespace("reticulate", quietly = TRUE) &&
        reticulate::py_module_available("transformers")
    }, error = function(e) FALSE),
    TRUE
  )
}

#' Get Feature Status
#'
#' @description
#' Returns availability status for all optional features.
#'
#' @return Named list with feature availability
#'
#' @export
#'
#' @examples
#' \donttest{
#' status <- get_feature_status()
#' print(status)
#' }
get_feature_status <- function() {
  features <- c("python", "ollama", "pdf_tables", "embeddings", "sentiment_deep")
  result <- lapply(features, function(f) {
    tryCatch(check_feature(f), error = function(e) FALSE)
  })
  names(result) <- features
  result$web <- check_web_deployment()
  result$local <- !check_web_deployment()
  return(result)
}

#' Show Web Deployment Banner
#'
#' @description
#' Creates a Shiny UI banner for web deployments showing feature limitations.
#'
#' @param disabled Character vector of disabled feature names (optional)
#'
#' @return A shiny tagList UI element (or NULL if local)
#'
#' @export
#'
#' @examples
#' \dontrun{
#' output$banner <- renderUI({ show_web_banner() })
#' }
show_web_banner <- function(disabled = NULL) {
  if (!check_web_deployment()) return(NULL)

  if (is.null(disabled)) {
    disabled <- c("Python PDF processing", "Local Ollama AI",
                  "Embedding analysis", "Large files (>10MB)")
  }

  feature_list <- paste0("<li>", disabled, "</li>", collapse = "")

  shiny::tagList(
    shiny::tags$details(
      class = "web-version-note",
      style = "margin: 10px; padding: 8px 12px; background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px; font-size: 14px;",
      shiny::tags$summary(
        style = "cursor: pointer; color: #1E40AF; font-weight: 500;",
        shiny::icon("info-circle"),
        " Web version - some features limited"
      ),
      shiny::tags$div(
        style = "margin-top: 8px; padding-left: 5px; color: #374151;",
        shiny::tags$p(
          style = "margin: 5px 0;",
          "For full features: ",
          shiny::tags$code(
            style = "background: #F3F4F6; padding: 2px 6px; border-radius: 3px; font-size: 13px;",
            "remotes::install_github('mshin77/TextAnalysisR')"
          )
        ),
        shiny::HTML(paste0("<ul style='margin: 5px 0; padding-left: 20px; color: #6B7280;'>", feature_list, "</ul>"))
      )
    )
  )
}

#' Require Feature
#'
#' @description
#' Checks feature availability and shows notification if unavailable.
#'
#' @param feature Character: feature name to check
#' @param session Shiny session object (optional)
#'
#' @return Logical TRUE if available, FALSE if not
#'
#' @export
#'
#' @examples
#' \dontrun{
#' if (!require_feature("embeddings", session)) return()
#' }
require_feature <- function(feature, session = NULL) {
  if (check_feature(feature)) return(TRUE)

  msg <- switch(feature,
    "python" = "Python not configured. Run setup_python_env().",
    "ollama" = "Ollama not available. Install from ollama.com.",
    "pdf_tables" = "PDF tables require Python. Run setup_python_env().",
    "embeddings" = "Embeddings require sentence-transformers. Run: pip install sentence-transformers torch",
    "sentiment_deep" = "Neural sentiment requires transformers. Run: pip install transformers torch",
    paste0("Feature '", feature, "' not available.")
  )

  if (check_web_deployment()) {
    msg <- paste(msg, "Only available in R package version.")
  }

  if (!is.null(session)) {
    shiny::showNotification(msg, type = "warning", duration = 8)
  } else {
    message(msg)
  }

  return(FALSE)
}

#' Create Formatted Analysis Data Table
#'
#' @description Creates a consistently formatted DT::datatable for analysis results with
#' export buttons and optional numeric formatting.
#'
#' @param data Data frame to display
#' @param colnames Optional character vector of column names for display
#' @param numeric_cols Optional character vector of numeric columns to round
#' @param digits Number of digits for rounding numeric columns (default: 3)
#' @param export_formats Character vector of export formats (default: c('copy', 'csv', 'excel', 'pdf', 'print'))
#' @param page_length Number of rows per page (default: 25)
#' @param font_size Font size for table cells (default: "16px")
#'
#' @return A DT::datatable object
#'
#' @export
#'
#' @examples
#' \dontrun{
#' df <- data.frame(term = c("word1", "word2"), score = c(0.123456, 0.789012))
#' create_analysis_datatable(df, numeric_cols = "score", digits = 3)
#' }
create_analysis_datatable <- function(data,
                                      colnames = NULL,
                                      numeric_cols = NULL,
                                      digits = 3,
                                      export_formats = c('copy', 'csv', 'excel', 'pdf', 'print'),
                                      page_length = 25,
                                      font_size = "16px") {

  dt <- DT::datatable(
    data,
    colnames = colnames,
    rownames = FALSE,
    extensions = 'Buttons',
    options = list(
      scrollX = TRUE,
      pageLength = page_length,
      dom = 'Bfrtip',
      buttons = export_formats
    )
  )

  if (!is.null(numeric_cols) && length(numeric_cols) > 0) {
    valid_cols <- intersect(numeric_cols, names(data))
    if (length(valid_cols) > 0) {
      dt <- DT::formatRound(dt, columns = valid_cols, digits = digits)
    }
  }

  dt <- DT::formatStyle(dt, columns = names(data), `font-size` = font_size)

  return(dt)
}

#' Calculate Cosine Similarity Matrix
#'
#' @description Calculates the cosine similarity between all pairs of rows in a matrix.
#' @param matrix_data A numeric matrix where rows represent documents/observations
#' @return A square similarity matrix with values between -1 and 1
#' @export
calculate_cosine_similarity <- function(matrix_data) {
  if (is.null(matrix_data) || nrow(matrix_data) == 0 || ncol(matrix_data) == 0) {
    return(matrix(1, nrow = 1, ncol = 1))
  }

  matrix_data[!is.finite(matrix_data)] <- 0

  normalized_matrix <- t(apply(matrix_data, 1, function(row) {
    norm <- sqrt(sum(row^2, na.rm = TRUE))
    if (norm > 0 && is.finite(norm)) {
      return(row / norm)
    } else {
      return(rep(0, length(row)))
    }
  }))

  normalized_matrix[!is.finite(normalized_matrix)] <- 0

  similarity_matrix <- normalized_matrix %*% t(normalized_matrix)

  similarity_matrix[!is.finite(similarity_matrix)] <- 0
  diag(similarity_matrix) <- 1
  similarity_matrix[similarity_matrix > 1] <- 1
  similarity_matrix[similarity_matrix < -1] <- -1

  return(similarity_matrix)
}

#' Create Error Plot for Plotly
#'
#' @description Creates a plotly error/status plot with a message
#' @param message The message to display
#' @param color Color for the text (default: "#ef4444")
#' @return A plotly plot object displaying the message
#' @export
plot_error <- function(message, color = "#ef4444") {
  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("plotly package is required for this function. ",
         "Please install it with: install.packages('plotly')")
  }

  plotly::plot_ly(type = "scatter", mode = "markers") %>%
    plotly::add_annotations(
      text = message,
      xref = "paper",
      yref = "paper",
      x = 0.5,
      y = 0.5,
      showarrow = FALSE,
      font = list(size = 14, color = color),
      xanchor = "center",
      yanchor = "middle"
    ) %>%
    plotly::layout(
      xaxis = list(visible = FALSE),
      yaxis = list(visible = FALSE),
      margin = list(t = 40, r = 40, b = 40, l = 40),
      plot_bgcolor = "white",
      paper_bgcolor = "white"
    )
}


#' @title Complete Text Mining Workflow
#'
#' @description
#' This function provides a complete text mining workflow that follows the same sequence
#' as the Shiny application: file processing → text uniting → preprocessing →
#' DFM creation → analysis.
#' It serves as a convenience function for users who want to execute the entire
#' pipeline programmatically.
#'
#' @param dataset_choice A character string indicating the dataset choice:
#'   "Upload an Example Dataset", "Upload Your File", "Copy and Paste Text".
#' @param file_info A data frame containing file information (for file upload).
#' @param text_input A character string containing text input (for copy-paste).
#' @param listed_vars A character vector of column names to unite into text.
#' @param min_char The minimum number of characters for tokens (default: 2).
#' @param remove_punct Logical; remove punctuation (default: TRUE).
#' @param remove_symbols Logical; remove symbols (default: TRUE).
#' @param remove_numbers Logical; remove numbers (default: TRUE).
#' @param remove_url Logical; remove URLs (default: TRUE).
#' @param detect_compounds Logical; detect multi-word expressions (default: FALSE).
#' @param compound_size Size range for compound detection (default: 2:3).
#' @param compound_min_count Minimum count for compounds (default: 2).
#' @param verbose Logical; print progress messages (default: TRUE).
#'
#' @return A list containing processed data, tokens, DFM, and metadata.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   # Using example dataset
#'   workflow_result <- TextAnalysisR::run_text_workflow(
#'     dataset_choice = "Upload an Example Dataset",
#'     listed_vars = c("title", "keyword", "abstract")
#'   )
#'
#'   # Using file upload
#'   file_info <- data.frame(filepath = "path/to/your/file.xlsx")
#'   workflow_result <- TextAnalysisR::run_text_workflow(
#'     dataset_choice = "Upload Your File",
#'     file_info = file_info,
#'     listed_vars = c("column1", "column2")
#'   )
#'
#'   # Using copy-paste text
#'   workflow_result <- TextAnalysisR::run_text_workflow(
#'     dataset_choice = "Copy and Paste Text",
#'     text_input = "Your text content here",
#'     listed_vars = "text"
#'   )
#' }
run_text_workflow <- function(dataset_choice,
                                         file_info = NULL,
                                         text_input = NULL,
                                         listed_vars,
                                         min_char = 2,
                                         remove_punct = TRUE,
                                         remove_symbols = TRUE,
                                         remove_numbers = TRUE,
                                         remove_url = TRUE,
                                         detect_compounds = FALSE,
                                         compound_size = 2:3,
                                         compound_min_count = 2,
                                         verbose = TRUE) {

  if (verbose) message("Starting complete text mining workflow...")
  start_time <- Sys.time()

  if (verbose) message("Step 1: Processing files...")
  processed_data <- import_files(
    dataset_choice = dataset_choice,
    file_info = file_info,
    text_input = text_input
  )

  if (verbose) message("Step 2: Uniting text columns...")
  united_data <- unite_cols(processed_data, listed_vars = listed_vars)

  if (verbose) message("Step 3: Preprocessing texts...")
  tokens <- prep_texts(
    united_tbl = united_data,
    text_field = "united_texts",
    min_char = min_char,
    remove_punct = remove_punct,
    remove_symbols = remove_symbols,
    remove_numbers = remove_numbers,
    remove_url = remove_url,
    verbose = verbose
  )

  compounds <- NULL
  if (detect_compounds) {
    if (verbose) message("Step 4: Detecting multi-word expressions...")
    compounds <- detect_multi_words(
      tokens = tokens,
      size = compound_size,
      min_count = compound_min_count
    )

    if (length(compounds) > 0) {
      tokens <- quanteda::tokens_compound(tokens, compounds)
    }
  }

  if (verbose) message("Step 5: Creating document-feature matrix...")
  dfm_object <- quanteda::dfm(tokens)

  execution_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  if (verbose) {
    message("Complete text mining workflow finished in ", round(execution_time, 2), " seconds")
    message("Documents processed: ", quanteda::ndoc(dfm_object))
    message("Features identified: ", quanteda::nfeat(dfm_object))
  }

  return(list(
    raw_data = processed_data,
    united_data = united_data,
    tokens = tokens,
    compounds = compounds,
    dfm = dfm_object,
    workflow_info = list(
      dataset_choice = dataset_choice,
      listed_vars = listed_vars,
      compounds_detected = !is.null(compounds),
      execution_time = execution_time,
      timestamp = Sys.time()
    )
  ))
}

#' @title Analyze and Visualize Word Frequencies Across a Continuous Variable
#'
#' @description
#' This function analyzes and visualizes word frequencies across a continuous variable.
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param continuous_variable A continuous variable in the metadata.
#' @param selected_terms A vector of terms to analyze trends for.
#' @param height The height of the resulting Plotly plot, in pixels (default: 500).
#' @param width The width of the resulting Plotly plot, in pixels (default: 900).
#'
#' @return A list containing Plotly objects and tables with the results.
#'
#' @details This function requires a fitted STM model object and a quanteda dfm object.
#' The continuous variable should be a column in the metadata of the dfm object.
#' The selected terms should be a vector of terms to analyze trends for.
#' The required packages are 'htmltools', 'splines', and 'broom' (plus
#' additional ones loaded internally).
#'
#' @importFrom stats glm reformulate binomial
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   mydata <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_cols(
#'     mydata,
#'     listed_vars = c("title", "keyword", "abstract")
#'   )
#'
#'   tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")
#'
#'   dfm_object <- quanteda::dfm(tokens)
#'
#'   word_freq_results <- TextAnalysisR::calculate_word_frequency(
#'     dfm_object,
#'     continuous_variable = "year",
#'     selected_terms = c("calculator", "computer"),
#'     height = 500,
#'     width = 900
#'   )
#'   print(word_freq_results$plot)
#'   print(word_freq_results$table)
#' }
calculate_word_frequency <- function(dfm_object,
                                 continuous_variable,
                                 selected_terms,
                                 height = 500,
                                 width = 900) {

  if (!requireNamespace("htmltools", quietly = TRUE) ||
      !requireNamespace("MASS", quietly = TRUE) ||
      !requireNamespace("pscl", quietly = TRUE) ||
      !requireNamespace("broom", quietly = TRUE)) {
    stop(
      "The 'htmltools', 'pscl', 'MASS', and 'broom' packages are required for this functionality. ",
      "Please install them using install.packages(c('htmltools', 'MASS', 'pscl', 'broom'))."
    )
  }

  dfm_outcome_obj <- dfm_object
  dfm_td <- tidytext::tidy(dfm_object)

  dfm_outcome_obj@docvars$document <- dfm_outcome_obj@docvars$docname_

  dfm_td <- dfm_td %>%
    left_join(dfm_outcome_obj@docvars,
              by = c("document" = "document"))

  con_var_term_counts <- dfm_td %>%
    tibble::as_tibble() %>%
    group_by(!!rlang::sym(continuous_variable)) %>%
    mutate(word_frequency = n()) %>%
    ungroup()

  con_var_term_gg <- con_var_term_counts %>%
    mutate(term = factor(term, levels = selected_terms)) %>%
    mutate(across(where(is.numeric), ~ round(., 3))) %>%
    filter(term %in% selected_terms) %>%
    ggplot(aes(
      x = !!rlang::sym(continuous_variable),
      y = word_frequency,
      group = term
    )) +
    geom_point(color = "#337ab7", alpha = 0.6, size = 1) +
    geom_line(color = "#337ab7", alpha = 0.6, linewidth = 0.5) +
    facet_wrap(~ term, scales = "free") +
    ggplot2::scale_y_continuous(labels = scales::number_format(accuracy = 1)) +
    labs(y = "Word Frequency") +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "none",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
      strip.text.x = element_text(size = 11, color = "#3B3B3B", face = "bold"),
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
      margin = list(l = 40, r = 150, t = 60, b = 40),
      xaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      hoverlabel = list(
        font = list(size = 16, family = "Roboto, sans-serif")
      )
    )

  significance_results <- con_var_term_counts %>%
    mutate(word = term) %>%
    filter(word %in% selected_terms) %>%
    group_by(word) %>%
    group_modify(~ {
      continuous_var <- if (is.null(continuous_variable) ||
                            length(continuous_variable) == 0) {
        stop("No continuous variable selected.")
      } else {
        continuous_variable[1]
      }

      df <- .x %>%
        dplyr::mutate(
          word_frequency = as.numeric(word_frequency),
          !!continuous_var := as.numeric(!!rlang::sym(continuous_var))
        ) %>%
        dplyr::filter(is.finite(word_frequency) &
                        is.finite(!!rlang::sym(continuous_var)))

      if (length(unique(df$word_frequency)) <= 1) {
        return(tibble::tibble(term = NA, estimate = NA, std.error = NA,
                              statistic = NA, p.value = NA,
                              `odds ratio` = NA, var.diag = NA,
                              `std.error (odds ratio)` = NA,
                              model_type = "Insufficient data"))
      }

      if (length(unique(df[[continuous_var]])) <= 1) {
        return(tibble::tibble(term = NA, estimate = NA, std.error = NA,
                              statistic = NA, p.value = NA,
                              `odds ratio` = NA, var.diag = NA,
                              `std.error (odds ratio)` = NA,
                              model_type = "Insufficient data"))
      }

      if (nrow(df) < 2) {
        return(tibble::tibble(term = NA, estimate = NA, std.error = NA,
                              statistic = NA, p.value = NA,
                              `odds ratio` = NA, var.diag = NA,
                              `std.error (odds ratio)` = NA,
                              model_type = "Insufficient data"))
      }

      formula_simple <- as.formula(paste0("word_frequency ~ ", continuous_var))

      mean_count <- mean(df$word_frequency, na.rm = TRUE)
      var_count <- var(df$word_frequency, na.rm = TRUE)
      dispersion_ratio <- ifelse(mean_count != 0, var_count / mean_count, NA)
      prop_zero <- mean(df$word_frequency == 0, na.rm = TRUE)

      model <- NULL

      if (prop_zero > 0.5) {
        model <- tryCatch(
          pscl::zeroinfl(formula_simple, data = df, dist = "negbin", link = "logit"),
          error = function(e) {
            return(NULL)
          }
        )
        if (!is.null(model)) {
          model_type <- "Zero-Inflated Negative Binomial"
        }
      }

      if (is.null(model)) {
        model <- tryCatch(
          MASS::glm.nb(formula_simple, data = df, control = glm.control(maxit = 200)),
          error = function(e) {
            return(NULL)
          }
        )
        if (is.null(model)) {
          model <- glm(formula_simple, family = poisson(link = "log"), data = df)
          model_type <- "Poisson"
        } else {
          model_type <- "Negative Binomial"
        }
      }

      tidy_result <- broom::tidy(model) %>%
        dplyr::mutate(
          `odds ratio` = exp(estimate),
          var.diag = diag(vcov(model)),
          `std.error (odds ratio)` = ifelse(var.diag >= 0,
                                            sqrt(`odds ratio`^2 * var.diag),
                                            NA),
          model_type = model_type
        )

      return(tidy_result)
    }) %>%
    ungroup() %>%
    dplyr::select(word, model_type, term, estimate, std.error,
                  `odds ratio`, `std.error (odds ratio)`, statistic, p.value) %>%
    rename(
      logit = estimate,
      `z-statistic` = statistic
    )


  significance_results_tables <- significance_results %>%
    mutate(word = factor(word, levels = selected_terms)) %>%
    arrange(word) %>%
    group_by(word) %>%
    group_map(~ {
      htmltools::tagList(
        htmltools::tags$div(
          style = "margin-bottom: 20px;",
          htmltools::tags$p(
            .y$word,
            style = "font-weight: bold; text-align: center; font-size: 11pt;"
          )
        ),
        .x %>%
          mutate_if(is.numeric, ~ round(., 3)) %>%
          DT::datatable(
            rownames = FALSE,
            extensions = 'Buttons',
            options = list(
              scrollX = TRUE,
              width = "80%",
              dom = 'Bfrtip',
              buttons = c('copy', 'csv', 'excel', 'pdf', 'print')
            )
          ) %>%
          DT::formatStyle(
            columns = names(.x),
            `font-size` = "16px"
          )
      )
    })

  list(
    plot = con_var_term_plotly,
    table = htmltools::tagList(significance_results_tables) %>% htmltools::browsable()
  )
}


#' @title Calculate Comprehensive Metrics
#'
#' @description
#' Calculates comprehensive similarity metrics including statistical measures
#' and network properties.
#' Internal function used by document_similarity_analysis.
#'
#' @param similarity_matrix A similarity matrix.
#' @param labels Optional vector of labels for clustering metrics.
#' @param method_info Optional method information.
#'
#' @return A list of comprehensive metrics.
#'
#' @keywords internal
calculate_metrics <- function(similarity_matrix, labels = NULL, method_info = NULL) {
  off_diagonal <- similarity_matrix[upper.tri(similarity_matrix) | lower.tri(similarity_matrix)]

  metrics <- list(
    n_docs = nrow(similarity_matrix),
    mean_similarity = round(mean(off_diagonal, na.rm = TRUE), 4),
    median_similarity = round(median(off_diagonal, na.rm = TRUE), 4),
    std_similarity = round(sd(off_diagonal, na.rm = TRUE), 4),
    min_similarity = round(min(off_diagonal, na.rm = TRUE), 4),
    max_similarity = round(max(off_diagonal, na.rm = TRUE), 4),
    similarity_range = paste(round(min(off_diagonal, na.rm = TRUE), 3), "to",
                             round(max(off_diagonal, na.rm = TRUE), 3)),
    sparsity = round(sum(off_diagonal < 0.1, na.rm = TRUE) / length(off_diagonal), 4),
    connectivity = round(sum(off_diagonal > 0.3, na.rm = TRUE) / length(off_diagonal), 4),
    skewness = ifelse(requireNamespace("moments", quietly = TRUE),
                     round(moments::skewness(off_diagonal, na.rm = TRUE), 4), NA),
    kurtosis = ifelse(requireNamespace("moments", quietly = TRUE),
                     round(moments::kurtosis(off_diagonal, na.rm = TRUE), 4), NA),
    silhouette_score = NA,
    modularity = NA
  )

  if (!is.null(labels) && is.atomic(labels) && is.vector(labels) &&
      length(labels) > 0 && length(unique(labels)) > 1 &&
      length(unique(labels)) < length(labels) * 0.8) {

    tryCatch({
      if (requireNamespace("cluster", quietly = TRUE)) {
        dist_matrix <- as.dist(1 - similarity_matrix)
        sil_result <- cluster::silhouette(as.numeric(as.factor(labels)), dist_matrix)
        metrics$silhouette_score <- if (is.numeric(sil_result[, 3]) && !is.na(sil_result[, 3]))
                                   round(mean(sil_result[, 3]), 3) else NA
      }
    }, error = function(e) {
      metrics$silhouette_score <- NA
    })

    tryCatch({
      if (requireNamespace("igraph", quietly = TRUE)) {
        threshold <- quantile(similarity_matrix[upper.tri(similarity_matrix)], 0.75, na.rm = TRUE)
        adj_matrix <- similarity_matrix > threshold
        diag(adj_matrix) <- FALSE
        graph <- igraph::graph_from_adjacency_matrix(adj_matrix, mode = "undirected")
        communities <- igraph::cluster_louvain(graph)
        metrics$modularity <- if (is.numeric(igraph::modularity(communities)) &&
                                  !is.na(igraph::modularity(communities)))
                             round(igraph::modularity(communities), 3) else NA
      }
    }, error = function(e) {
      metrics$modularity <- NA
    })
  }


  if (!is.null(method_info)) {
    metrics$method <- method_info$method
    metrics$model_name <- method_info$model_name %||% "N/A"
  }

  return(metrics)
}


################################################################################
# PYTHON ENVIRONMENT SETUP AND UTILITIES
################################################################################

#' Setup Python Environment
#'
#' @description
#' Intelligently sets up Python virtual environment with required packages.
#' Detects existing Python installations and guides users if Python is missing.
#'
#' @param envname Character string name for the virtual environment
#'   (default: "textanalysisr-env")
#' @param force Logical, whether to recreate environment if it exists
#'   (default: FALSE)
#'
#' @return Invisible TRUE if successful, stops with error message if failed
#'
#' @details
#' This function:
#' - Automatically detects if Python is already installed
#' - Offers to install Miniconda if no Python found
#' - Creates an isolated virtual environment (does NOT modify system Python)
#' - Installs minimal core packages:
#'   * spacy (NLP processing)
#'   * pdfplumber (PDF table extraction)
#' - Dependencies installed automatically by pip
#' - Avoids heavy packages (no torch, transformers)
#'
#' The virtual environment approach means:
#' - No conflicts with other Python projects
#' - Easy to remove (just delete the environment)
#' - System Python remains untouched
#' - Much smaller download (~100MB vs 5GB+)
#'
#' After setup, restart R session to activate enhanced features.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # First time setup (auto-detects Python)
#' setup_python_env()
#'
#' # Recreate environment
#' setup_python_env(force = TRUE)
#' }
setup_python_env <- function(envname = "textanalysisr-env", force = FALSE) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it with: install.packages('reticulate')")
  }

  message("\nPython Environment Setup")


  # Check if Python is available
  python_available <- tryCatch({
    py_config <- reticulate::py_discover_config()
    !is.null(py_config$python)
  }, error = function(e) FALSE)

  if (!python_available) {
    message("No Python found. Install Miniconda? (y/n): ")
    response <- readline(prompt = "")

    if (tolower(trimws(response)) == "y") {
      message("Installing Miniconda...")
      reticulate::install_miniconda()
      message("Done.")
    } else {
      stop("Python required. Install from python.org and retry.")
    }
  } else {
    py_info <- reticulate::py_discover_config()
    message("Python: ", py_info$python, " (v", py_info$version, ")")
  }

  tryCatch({
    env_exists <- envname %in% reticulate::virtualenv_list()

    if (env_exists && !force) {
      message("Environment '", envname, "' exists. Use force=TRUE to recreate.")
      reticulate::use_virtualenv(envname, required = TRUE)
      return(invisible(TRUE))
    }

    if (env_exists && force) {
      message("Removing existing environment...")
      reticulate::virtualenv_remove(envname, confirm = FALSE)
    }

    message("Creating environment '", envname, "'...")
    reticulate::virtualenv_create(envname, python = NULL)
    reticulate::use_virtualenv(envname, required = TRUE)

    requirements_file <- system.file("python", "requirements.txt", package = "TextAnalysisR")

    message("Installing packages...")
    if (file.exists(requirements_file)) {
      req_packages <- readLines(requirements_file)
      req_packages <- req_packages[!grepl("^#|^\\s*$", req_packages)]
      reticulate::virtualenv_install(envname = envname, packages = req_packages, ignore_installed = FALSE)
    } else {
      packages <- c(
        "spacy>=3.5.0",
        "pdfplumber>=0.10.0"
      )
      reticulate::virtualenv_install(envname = envname, packages = packages, ignore_installed = FALSE)
    }

    message("Testing imports...")
    test_result <- tryCatch({
      reticulate::py_run_string("import spacy")
      reticulate::py_run_string("import pdfplumber")
      TRUE
    }, error = function(e) {
      message("Import failed: ", e$message)
      FALSE
    })

    if (!test_result) {
      stop("Package imports failed. Check Python logs.")
    }

    # Download spaCy English model
    message("Downloading spaCy English model (en_core_web_sm)...")
    spacy_model_result <- tryCatch({
      reticulate::py_run_string("
import spacy
try:
    nlp = spacy.load('en_core_web_sm')
    print('Model already installed')
except OSError:
    print('Downloading model...')
    spacy.cli.download('en_core_web_sm')
    print('Model downloaded')
")
      TRUE
    }, error = function(e) {
      message("Note: spaCy model download failed: ", e$message)
      message("You can install manually: python -m spacy download en_core_web_sm")
      FALSE
    })

    message("\nSetup complete!")
    if (spacy_model_result) {
      message("- spaCy model: en_core_web_sm installed")
    }
    message("Restart R session to activate.")
    return(invisible(TRUE))

  }, error = function(e) {
    stop("Failed to set up Python environment: ", e$message)
  })
}


#' Check Python Environment Status
#'
#' @description
#' Checks if Python environment is available and properly configured.
#'
#' @param envname Character string name of the virtual environment
#'   (default: "textanalysisr-env")
#'
#' @return List with status information:
#'   - available: Logical, TRUE if environment exists
#'   - active: Logical, TRUE if environment is currently active
#'   - packages: List of installed package versions
#'
#' @export
#'
#' @examples
#' \dontrun{
#' status <- check_python_env()
#' print(status)
#' }
check_python_env <- function(envname = "textanalysisr-env") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }

  env_list <- reticulate::virtualenv_list()
  available <- envname %in% env_list

  if (!available) {
    return(list(
      available = FALSE,
      active = FALSE,
      packages = NULL,
      message = paste("Environment", envname, "not found. Run setup_python_env() to create it.")
    ))
  }

  tryCatch({
    reticulate::use_virtualenv(envname, required = TRUE)

    packages <- tryCatch({
      spacy_version <- reticulate::py_run_string("import spacy; print(spacy.__version__)")
      pdfplumber_check <- reticulate::py_run_string("import pdfplumber; print(pdfplumber.__version__)")

      list(
        spacy = spacy_version,
        pdfplumber = pdfplumber_check
      )
    }, error = function(e) NULL)

    return(list(
      available = TRUE,
      active = TRUE,
      packages = packages,
      message = "Python environment is ready"
    ))

  }, error = function(e) {
    return(list(
      available = TRUE,
      active = FALSE,
      packages = NULL,
      message = paste("Failed to activate environment:", e$message)
    ))
  })
}




################################################################################
# OLLAMA LOCAL LLM UTILITIES
################################################################################

#' Check if Ollama is Available
#'
#' @description Checks if Ollama is installed and running on the local machine.
#'
#' @param verbose Logical, if TRUE, prints status messages.
#'
#' @return Logical indicating whether Ollama is available.
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' if (check_ollama()) {
#'   message("Ollama is ready!")
#' }
#' }
check_ollama <- function(verbose = FALSE) {
  tryCatch({
    response <- httr::GET(
      "http://localhost:11434/api/tags",
      httr::timeout(2)
    )

    is_available <- httr::status_code(response) == 200

    if (verbose) {
      if (is_available) {
        message("Ollama is available and running")
      } else {
        message("Ollama server responded but returned error status")
      }
    }

    return(is_available)

  }, error = function(e) {
    if (verbose) {
      message("Ollama is not available: ", e$message)
      message("To use Ollama, please install it from https://ollama.com")
    }
    return(FALSE)
  })
}

#' List Available Ollama Models
#'
#' @description Lists all models currently installed in Ollama.
#'
#' @param verbose Logical, if TRUE, prints status messages.
#'
#' @return Character vector of model names, or NULL if Ollama is unavailable.
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' models <- list_ollama_models()
#' print(models)
#' }
list_ollama_models <- function(verbose = FALSE) {
  if (!check_ollama(verbose = FALSE)) {
    if (verbose) {
      message("Ollama is not available")
    }
    return(NULL)
  }

  tryCatch({
    response <- httr::GET(
      "http://localhost:11434/api/tags",
      httr::timeout(5)
    )

    if (httr::status_code(response) == 200) {
      content <- jsonlite::fromJSON(httr::content(response, "text", encoding = "UTF-8"))

      if (!is.null(content$models) && length(content$models) > 0) {
        model_names <- content$models$name
        if (verbose) {
          message("Found ", length(model_names), " Ollama models:")
          for (model in model_names) {
            message("  - ", model)
          }
        }
        return(model_names)
      } else {
        if (verbose) {
          message("No Ollama models found. Please pull a model:")
          message("  ollama pull llama3.2")
        }
        return(character(0))
      }
    }

    return(NULL)

  }, error = function(e) {
    if (verbose) {
      message("Error listing Ollama models: ", e$message)
    }
    return(NULL)
  })
}

#' Call Ollama for Text Generation
#'
#' @description Sends a prompt to Ollama and returns the generated text.
#'
#' @param prompt Character string containing the prompt.
#' @param model Character string specifying the Ollama model (default: "llama3.2").
#' @param temperature Numeric value controlling randomness (default: 0.3).
#' @param max_tokens Maximum number of tokens to generate (default: 512).
#' @param timeout Timeout in seconds for the request (default: 60).
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return Character string with the generated text, or NULL if failed.
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' response <- call_ollama(
#'   prompt = "Summarize these keywords: machine learning, neural networks, AI",
#'   model = "llama3.2"
#' )
#' print(response)
#' }
call_ollama <- function(prompt,
                       model = "llama3.2",
                       temperature = 0.3,
                       max_tokens = 512,
                       timeout = 60,
                       verbose = FALSE) {

  if (!check_ollama(verbose = verbose)) {
    stop("Ollama is not available. Please ensure Ollama is installed and running.")
  }

  if (verbose) {
    message("Calling Ollama with model: ", model)
  }

  tryCatch({
    body <- list(
      model = model,
      prompt = prompt,
      stream = FALSE,
      options = list(
        temperature = temperature,
        num_predict = max_tokens
      )
    )

    response <- httr::POST(
      "http://localhost:11434/api/generate",
      body = jsonlite::toJSON(body, auto_unbox = TRUE),
      httr::content_type_json(),
      httr::timeout(timeout)
    )

    if (httr::status_code(response) == 200) {
      content <- jsonlite::fromJSON(httr::content(response, "text", encoding = "UTF-8"))

      if (!is.null(content$response)) {
        if (verbose) {
          message("Ollama response received successfully")
        }
        return(trimws(content$response))
      } else {
        warning("Ollama response was empty")
        return(NULL)
      }
    } else {
      warning("Ollama API returned status code: ", httr::status_code(response))
      return(NULL)
    }

  }, error = function(e) {
    warning("Error calling Ollama: ", e$message)
    return(NULL)
  })
}

#' Get Recommended Ollama Model
#'
#' @description Returns a recommended Ollama model based on what's available.
#'
#' @param preferred_models Character vector of preferred models in priority order.
#' @param verbose Logical, if TRUE, prints status messages.
#'
#' @return Character string of recommended model, or NULL if none available.
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' model <- get_recommended_ollama_model()
#' print(model)
#' }
get_recommended_ollama_model <- function(preferred_models = c("llama3.2", "gemma3", "mistral:7b", "tinyllama"),
                                        verbose = FALSE) {

  available_models <- list_ollama_models(verbose = FALSE)

  if (is.null(available_models) || length(available_models) == 0) {
    if (verbose) {
      message("No Ollama models available. Recommended models:")
      message("  1. llama3.2 (2.0GB, best balance)")
      message("  2. gemma3 (3.9GB, strong reasoning)")
      message("  3. mistral:7b (4.1GB, high quality)")
      message("\nTo install: ollama pull llama3.2")
    }
    return(NULL)
  }

  for (preferred in preferred_models) {
    if (preferred %in% available_models) {
      if (verbose) {
        message("Using model: ", preferred)
      }
      return(preferred)
    }
  }

  if (verbose) {
    message("Using first available model: ", available_models[1])
  }
  return(available_models[1])
}


################################################################################
# CLOUD LLM API UTILITIES (OpenAI, Gemini)
################################################################################

#' Call OpenAI Chat Completion API
#'
#' @description
#' Makes a chat completion request to OpenAI's API.
#'
#' @param system_prompt Character string with system instructions
#' @param user_prompt Character string with user message
#' @param model Character string specifying the model (default: "gpt-4.1-mini")
#' @param temperature Numeric temperature for response randomness (default: 0)
#' @param max_tokens Maximum number of tokens to generate (default: 150)
#' @param api_key Character string with OpenAI API key
#'
#' @return Character string with the model's response
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' response <- call_openai_chat(
#'   system_prompt = "You are a helpful assistant.",
#'   user_prompt = "Generate a topic label for: education, student, learning",
#'   api_key = Sys.getenv("OPENAI_API_KEY")
#' )
#' }
call_openai_chat <- function(system_prompt,
                              user_prompt,
                              model = "gpt-4.1-mini",
                              temperature = 0,
                              max_tokens = 150,
                              api_key) {

  if (!requireNamespace("httr", quietly = TRUE)) {
    stop("httr package is required for OpenAI API calls")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite package is required for OpenAI API calls")
  }

  body_list <- list(
    model = model,
    messages = list(
      list(role = "system", content = system_prompt),
      list(role = "user", content = user_prompt)
    ),
    temperature = temperature,
    max_tokens = max_tokens
  )

  response <- httr::POST(
    url = "https://api.openai.com/v1/chat/completions",
    httr::add_headers(
      `Content-Type` = "application/json",
      `Authorization` = paste("Bearer", api_key)
    ),
    body = jsonlite::toJSON(body_list, auto_unbox = TRUE),
    encode = "json"
  )

  if (httr::status_code(response) != 200) {
    error_content <- httr::content(response, "text", encoding = "UTF-8")
    stop(sprintf("OpenAI API error (status %d): %s",
                 httr::status_code(response), error_content))
  }

  res_json <- jsonlite::fromJSON(httr::content(response, "text", encoding = "UTF-8"))

  if (!is.null(res_json$choices) && length(res_json$choices) > 0) {
    return(res_json$choices$message$content[1])
  }

  stop("Unexpected response structure from OpenAI API")
}


#' Call Gemini Chat API
#'
#' @description
#' Makes a chat completion request to Google's Gemini API.
#'
#' @param system_prompt Character string with system instructions
#' @param user_prompt Character string with user message
#' @param model Character string specifying the Gemini model (default: "gemini-2.5-flash")
#' @param temperature Numeric temperature for response randomness (default: 0)
#' @param max_tokens Maximum number of tokens to generate (default: 150)
#' @param api_key Character string with Gemini API key
#'
#' @return Character string with the model's response
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' response <- call_gemini_chat(
#'   system_prompt = "You are a helpful assistant.",
#'   user_prompt = "Generate a topic label for: education, student, learning",
#'   api_key = Sys.getenv("GEMINI_API_KEY")
#' )
#' }
call_gemini_chat <- function(system_prompt,
                              user_prompt,
                              model = "gemini-2.5-flash",
                              temperature = 0,
                              max_tokens = 150,
                              api_key) {

  if (!requireNamespace("httr", quietly = TRUE)) {
    stop("httr package is required for Gemini API calls")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite package is required for Gemini API calls")
  }

  # Gemini uses a different message format
  body_list <- list(
    contents = list(
      list(
        role = "user",
        parts = list(
          list(text = paste0(system_prompt, "\n\n", user_prompt))
        )
      )
    ),
    generationConfig = list(
      temperature = temperature,
      maxOutputTokens = max_tokens
    )
  )

  url <- paste0(
    "https://generativelanguage.googleapis.com/v1beta/models/",
    model, ":generateContent?key=", api_key
  )

  response <- httr::POST(
    url = url,
    httr::add_headers(`Content-Type` = "application/json"),
    body = jsonlite::toJSON(body_list, auto_unbox = TRUE),
    encode = "json"
  )

  if (httr::status_code(response) != 200) {
    error_content <- httr::content(response, "text", encoding = "UTF-8")
    stop(sprintf("Gemini API error (status %d): %s",
                 httr::status_code(response), error_content))
  }

  res_json <- jsonlite::fromJSON(httr::content(response, "text", encoding = "UTF-8"))

  if (!is.null(res_json$candidates) && length(res_json$candidates) > 0) {
    # Extract text from Gemini response structure
    parts <- res_json$candidates[[1]]$content$parts
    if (!is.null(parts) && length(parts) > 0) {
      return(parts[[1]]$text)
    }
  }

  stop("Unexpected response structure from Gemini API")
}


#' Call LLM API (Unified Wrapper)
#'
#' @description
#' Unified wrapper for calling different LLM providers (OpenAI, Gemini, Ollama).
#' Automatically routes to the appropriate provider-specific function.
#'
#' @param provider Character string: "openai", "gemini", or "ollama"
#' @param system_prompt Character string with system instructions
#' @param user_prompt Character string with user message
#' @param model Character string specifying the model (provider-specific defaults apply)
#' @param temperature Numeric temperature for response randomness (default: 0)
#' @param max_tokens Maximum number of tokens to generate (default: 150)
#' @param api_key Character string with API key (required for openai/gemini)
#'
#' @return Character string with the model's response
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' # Using OpenAI
#' response <- call_llm_api(
#'   provider = "openai",
#'   system_prompt = "You are a helpful assistant.",
#'   user_prompt = "Generate a topic label",
#'   api_key = Sys.getenv("OPENAI_API_KEY")
#' )
#'
#' # Using Gemini
#' response <- call_llm_api(
#'   provider = "gemini",
#'   system_prompt = "You are a helpful assistant.",
#'   user_prompt = "Generate a topic label",
#'   api_key = Sys.getenv("GEMINI_API_KEY")
#' )
#' }
call_llm_api <- function(provider = c("openai", "gemini", "ollama"),
                         system_prompt,
                         user_prompt,
                         model = NULL,
                         temperature = 0,
                         max_tokens = 150,
                         api_key = NULL) {

  provider <- match.arg(provider)

  # Set default models based on provider
  if (is.null(model)) {
    model <- switch(provider,
      "openai" = "gpt-4.1-mini",
      "gemini" = "gemini-2.5-flash",
      "ollama" = "llama3.2"
    )
  }

  # Validate API key for cloud providers
  if (provider %in% c("openai", "gemini")) {
    if (is.null(api_key) || !nzchar(api_key)) {
      # Try to get from environment
      api_key <- switch(provider,
        "openai" = Sys.getenv("OPENAI_API_KEY"),
        "gemini" = Sys.getenv("GEMINI_API_KEY")
      )
    }

    if (!nzchar(api_key)) {
      stop(paste0("API key required for ", provider, ". Set ", toupper(provider), "_API_KEY environment variable."))
    }
  }

  # Route to appropriate provider
  result <- switch(provider,
    "openai" = call_openai_chat(
      system_prompt = system_prompt,
      user_prompt = user_prompt,
      model = model,
      temperature = temperature,
      max_tokens = max_tokens,
      api_key = api_key
    ),
    "gemini" = call_gemini_chat(
      system_prompt = system_prompt,
      user_prompt = user_prompt,
      model = model,
      temperature = temperature,
      max_tokens = max_tokens,
      api_key = api_key
    ),
    "ollama" = call_ollama(
      prompt = paste0(system_prompt, "\n\n", user_prompt),
      model = model,
      temperature = temperature,
      max_tokens = max_tokens
    )
  )

  return(result)
}


#' Get Embeddings from API
#'
#' @description
#' Generates text embeddings using Ollama (local), OpenAI, or Gemini embedding APIs.
#'
#' @param texts Character vector of texts to embed
#' @param provider Character string: "ollama" (local API, free), "openai", or "gemini"
#' @param model Character string specifying the embedding model. Defaults:
#'   - ollama: "nomic-embed-text"
#'   - openai: "text-embedding-3-small"
#'   - gemini: "gemini-embedding-001"
#' @param api_key Character string with API key (not required for Ollama)
#' @param batch_size Integer, number of texts to embed per API call (default: 100)
#'
#' @return Matrix with embeddings (rows = texts, columns = dimensions)
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' data(SpecialEduTech)
#' texts <- SpecialEduTech$abstract[1:5]
#'
#' # Using local Ollama API (free, no API key required)
#' embeddings <- get_api_embeddings(texts, provider = "ollama")
#'
#' # Using OpenAI API
#' embeddings <- get_api_embeddings(texts, provider = "openai")
#'
#' dim(embeddings)
#' }
get_api_embeddings <- function(texts,
                           provider = c("ollama", "openai", "gemini"),
                           model = NULL,
                           api_key = NULL,
                           batch_size = 100) {

  provider <- match.arg(provider)

  # Set default models
  if (is.null(model)) {
    model <- switch(provider,
      "ollama" = "nomic-embed-text",
      "openai" = "text-embedding-3-small",
      "gemini" = "gemini-embedding-001"
    )
  }

  # Get API key from environment if not provided (not needed for Ollama)
  if (provider != "ollama") {
    if (is.null(api_key) || !nzchar(api_key)) {
      api_key <- switch(provider,
        "openai" = Sys.getenv("OPENAI_API_KEY"),
        "gemini" = Sys.getenv("GEMINI_API_KEY")
      )
    }

    if (!nzchar(api_key)) {
      stop(paste0("API key required. Set ", toupper(provider), "_API_KEY environment variable."))
    }
  }

  if (!requireNamespace("httr", quietly = TRUE)) {
    stop("httr package is required for embedding API calls")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite package is required for embedding API calls")
  }

  # Process in batches
  n_texts <- length(texts)
  all_embeddings <- list()

  for (start in seq(1, n_texts, by = batch_size)) {
    end <- min(start + batch_size - 1, n_texts)
    batch_texts <- texts[start:end]

    embeddings <- switch(provider,
      "ollama" = get_ollama_embeddings(batch_texts, model),
      "openai" = get_openai_embeddings(batch_texts, model, api_key),
      "gemini" = get_gemini_embeddings(batch_texts, model, api_key)
    )

    all_embeddings[[length(all_embeddings) + 1]] <- embeddings
  }

  # Combine all batches
  do.call(rbind, all_embeddings)
}


#' Get OpenAI Embeddings (Internal)
#' @keywords internal
get_openai_embeddings <- function(texts, model, api_key) {
  body_list <- list(
    input = texts,
    model = model
  )

  response <- httr::POST(
    url = "https://api.openai.com/v1/embeddings",
    httr::add_headers(
      `Content-Type` = "application/json",
      `Authorization` = paste("Bearer", api_key)
    ),
    body = jsonlite::toJSON(body_list, auto_unbox = TRUE),
    encode = "json"
  )

  if (httr::status_code(response) != 200) {
    error_content <- httr::content(response, "text", encoding = "UTF-8")
    stop(sprintf("OpenAI Embeddings API error (status %d): %s",
                 httr::status_code(response), error_content))
  }

  res_json <- jsonlite::fromJSON(httr::content(response, "text", encoding = "UTF-8"))

  if (!is.null(res_json$data)) {
    # Sort by index to ensure correct order
    embeddings <- res_json$data[order(res_json$data$index), ]
    do.call(rbind, embeddings$embedding)
  } else {
    stop("Unexpected response structure from OpenAI Embeddings API")
  }
}


#' Get Gemini Embeddings (Internal)
#' @keywords internal
get_gemini_embeddings <- function(texts, model, api_key) {
  # Gemini requires individual requests per text
  embeddings_list <- lapply(texts, function(text) {
    body_list <- list(
      model = paste0("models/", model),
      content = list(
        parts = list(
          list(text = text)
        )
      )
    )

    url <- paste0(
      "https://generativelanguage.googleapis.com/v1beta/models/",
      model, ":embedContent?key=", api_key
    )

    response <- httr::POST(
      url = url,
      httr::add_headers(`Content-Type` = "application/json"),
      body = jsonlite::toJSON(body_list, auto_unbox = TRUE),
      encode = "json"
    )

    if (httr::status_code(response) != 200) {
      error_content <- httr::content(response, "text", encoding = "UTF-8")
      stop(sprintf("Gemini Embeddings API error (status %d): %s",
                   httr::status_code(response), error_content))
    }

    res_json <- jsonlite::fromJSON(httr::content(response, "text", encoding = "UTF-8"))

    if (!is.null(res_json$embedding$values)) {
      return(res_json$embedding$values)
    } else {
      stop("Unexpected response structure from Gemini Embeddings API")
    }
  })

  do.call(rbind, embeddings_list)
}


#' Get Ollama Embeddings (Internal)
#'
#' Generate embeddings using local Ollama models.
#'
#' @param texts Character vector of texts to embed.
#' @param model Ollama embedding model (default: "nomic-embed-text").
#' @return Numeric matrix of embeddings (one row per text).
#' @keywords internal
get_ollama_embeddings <- function(texts, model = "nomic-embed-text") {
  if (!check_ollama(verbose = FALSE)) {
    stop("Ollama is not running. Start Ollama and try again.")
  }

  embeddings_list <- lapply(texts, function(text) {
    body <- list(
      model = model,
      prompt = text
    )

    response <- tryCatch({
      httr::POST(
        url = "http://localhost:11434/api/embeddings",
        body = jsonlite::toJSON(body, auto_unbox = TRUE),
        httr::content_type_json(),
        httr::timeout(60)
      )
    }, error = function(e) {
      stop(paste("Ollama embeddings request failed:", e$message))
    })

    if (httr::status_code(response) != 200) {
      error_content <- httr::content(response, "text", encoding = "UTF-8")
      stop(sprintf("Ollama embeddings error (status %d): %s",
                   httr::status_code(response), error_content))
    }

    res_json <- jsonlite::fromJSON(httr::content(response, "text", encoding = "UTF-8"))

    if (!is.null(res_json$embedding)) {
      return(res_json$embedding)
    } else {
      stop("Unexpected response structure from Ollama embeddings API")
    }
  })

  do.call(rbind, embeddings_list)
}


#' Get Best Available Embeddings
#'
#' @description
#' Auto-detects and uses the best available embedding provider with the following priority:
#' 1. Ollama (free, local, fast) - if running
#' 2. sentence-transformers (local Python) - if Python environment is set up
#' 3. OpenAI API - if OPENAI_API_KEY is set
#' 4. Gemini API - if GEMINI_API_KEY is set
#'
#' @param texts Character vector of texts to embed
#' @param provider Character string: "auto" (default), "ollama", "sentence-transformers",
#'   "openai", or "gemini". Use "auto" for automatic detection.
#' @param model Character string specifying the embedding model. If NULL, uses default
#'   model for the selected provider.
#' @param api_key Optional API key for OpenAI or Gemini providers. If NULL, falls back
#'   to environment variables (OPENAI_API_KEY, GEMINI_API_KEY).
#' @param verbose Logical, whether to print progress messages (default: TRUE)
#'
#' @return Matrix with embeddings (rows = texts, columns = dimensions)
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' data(SpecialEduTech)
#' texts <- SpecialEduTech$abstract[1:5]
#'
#' # Auto-detect best available provider
#' embeddings <- get_best_embeddings(texts)
#'
#' # Force specific provider
#' embeddings <- get_best_embeddings(texts, provider = "ollama")
#'
#' dim(embeddings)
#' }
get_best_embeddings <- function(texts,
                                 provider = "auto",
                                 model = NULL,
                                 api_key = NULL,
                                 verbose = TRUE) {

  if (!is.character(texts) || length(texts) == 0) {
    stop("texts must be a non-empty character vector")
  }

  if (provider == "auto") {
    if (check_ollama(verbose = FALSE)) {
      provider <- "ollama"
      if (verbose) message("Using Ollama embeddings (local)")
    } else if (check_feature("python")) {
      provider <- "sentence-transformers"
      if (verbose) message("Using sentence-transformers embeddings (local Python)")
    } else if ((!is.null(api_key) && nzchar(api_key)) || nzchar(Sys.getenv("OPENAI_API_KEY"))) {
      provider <- "openai"
      if (verbose) message("Using OpenAI embeddings (API)")
    } else if (nzchar(Sys.getenv("GEMINI_API_KEY"))) {
      provider <- "gemini"
      if (verbose) message("Using Gemini embeddings (API)")
    } else {
      stop(paste0(
        "No embedding provider available. Options:\n",
        "  1. Install and start Ollama (https://ollama.com)\n",
        "  2. Set up Python with: TextAnalysisR::setup_python_env()\n",
        "  3. Set OPENAI_API_KEY or GEMINI_API_KEY environment variable"
      ))
    }
  }

  embeddings <- switch(provider,
    "ollama" = {
      if (!check_ollama(verbose = FALSE)) {
        stop("Ollama is not running. Start Ollama and try again.")
      }
      model <- model %||% "nomic-embed-text"
      get_api_embeddings(texts, provider = "ollama", model = model)
    },
    "sentence-transformers" = {
      if (!check_feature("python")) {
        stop("Python not available. Run TextAnalysisR::setup_python_env() first.")
      }
      model <- model %||% "all-MiniLM-L6-v2"
      generate_embeddings(texts, model = model, verbose = verbose)
    },
    "openai" = {
      model <- model %||% "text-embedding-3-small"
      get_api_embeddings(texts, provider = "openai", model = model, api_key = api_key)
    },
    "gemini" = {
      model <- model %||% "gemini-embedding-001"
      get_api_embeddings(texts, provider = "gemini", model = model, api_key = api_key)
    },
    stop(paste0(
      "Unknown provider: ", provider, ". ",
      "Valid options: auto, ollama, sentence-transformers, openai, gemini"
    ))
  )

  embeddings
}


################################################################################
# SECURITY AND VALIDATION UTILITIES
################################################################################

#' Cybersecurity Utility Functions
#'
#' @description Functions for input validation, sanitization, and security logging
#'
#' @section NIST Compliance:
#' This package follows NIST security standards (based on NIST SP 800-53):
#' - SC-8: Transmission Confidentiality and Integrity (HTTPS encryption)
#' - SC-28: Protection of Information at Rest (secure API key storage)
#' - IA-5: Authenticator Management (API key validation and format checking)
#' - AC-3: Access Enforcement (rate limiting, input validation, file type restrictions)
#' - SI-10: Information Input Validation (malicious content detection)
#' - AU-2: Audit Events (security logging and monitoring)

#' Validate File Upload
#'
#' @param file_info File info object from Shiny fileInput
#' @return TRUE if valid, stops with error message if invalid
#' @keywords internal
validate_file_upload <- function(file_info) {
  if (is.null(file_info)) {
    stop("No file provided")
  }

  allowed_extensions <- c(".csv", ".xlsx", ".txt", ".rds", ".pdf", ".docx")
  ext <- tools::file_ext(file_info$name)

  if (!paste0(".", tolower(ext)) %in% allowed_extensions) {
    stop("Invalid file type. Allowed types: CSV, XLSX, TXT, RDS, PDF, DOCX")
  }

  max_size <- 50 * 1024 * 1024
  if (file_info$size > max_size) {
    stop("File size exceeds maximum limit of 50MB")
  }

  if (ext %in% c("csv", "txt")) {
    content <- tryCatch({
      suppressWarnings(readLines(file_info$datapath, n = 10, warn = FALSE))
    }, error = function(e) {
      stop("Unable to read file contents")
    })

    suspicious_patterns <- c("<script", "javascript:", "onerror=", "onclick=")
    if (any(grepl(paste(suspicious_patterns, collapse = "|"), content, ignore.case = TRUE))) {
      stop("File contains potentially malicious content")
    }
  }

  return(TRUE)
}

#' Sanitize Text Input
#'
#' @param text Text input from user
#' @return Sanitized text
#' @keywords internal
sanitize_text_input <- function(text) {
  if (is.null(text) || nchar(text) == 0) {
    return(text)
  }

  text <- gsub("<script.*?>.*?</script>", "", text, ignore.case = TRUE)
  text <- gsub("javascript:", "", text, ignore.case = TRUE)
  text <- gsub("on\\w+\\s*=", "", text, ignore.case = TRUE)

  max_chars <- 1000000
  if (nchar(text) > max_chars) {
    stop("Text input exceeds maximum length of 1 million characters")
  }

  return(text)
}

#' Check Rate Limit
#'
#' @param session_token Shiny session token
#' @param user_requests Reactive value storing request history
#' @param max_requests Maximum requests allowed in time window
#' @param window_seconds Time window in seconds (default: 3600 = 1 hour)
#' @return TRUE if within limit, stops with error if exceeded
#' @keywords internal
check_rate_limit <- function(session_token, user_requests, max_requests = 100, window_seconds = 3600) {
  current_time <- Sys.time()
  requests <- user_requests()

  if (is.null(requests[[session_token]])) {
    requests[[session_token]] <- list()
  }

  requests[[session_token]] <- Filter(function(x) {
    difftime(current_time, x, units = "secs") < window_seconds
  }, requests[[session_token]])

  if (length(requests[[session_token]]) >= max_requests) {
    stop("Rate limit exceeded. You have made too many requests. Please wait before trying again.")
  }

  requests[[session_token]] <- c(requests[[session_token]], current_time)
  user_requests(requests)

  return(TRUE)
}

#' Log Security Event
#'
#' @param event_type Type of security event
#' @param details Additional details about the event
#' @param session_info Shiny session object
#' @param level Log level (INFO, WARNING, ERROR)
#' @keywords internal
log_security_event <- function(event_type, details, session_info, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")

  session_token <- if (!is.null(session_info$token)) {
    substr(session_info$token, 1, 8)
  } else {
    "unknown"
  }

  log_message <- paste0(
    "[", timestamp, "] ",
    "[", level, "] ",
    "SECURITY: ", event_type, " | ",
    "Session: ", session_token, "... | ",
    "Details: ", details
  )

  log_file <- "security.log"
  tryCatch({
    cat(log_message, "\n", file = log_file, append = TRUE)
  }, error = function(e) {
    warning("Failed to write security log: ", e$message)
  })

  if (Sys.getenv("ENVIRONMENT") != "production") {
    message(log_message)
  }

  return(invisible(NULL))
}

#' Validate API Key Format
#'
#' @description
#' Validates API key format for OpenAI or Gemini according to NIST IA-5(1).
#' Auto-detects provider from key prefix and validates format requirements.
#'
#' @param api_key Character string containing the API key
#' @param strict Logical, if TRUE performs additional validation checks
#'
#' @return List with valid (logical), provider (character), and error (character if invalid)
#' @keywords internal
#'
#' @section NIST Compliance:
#' Implements NIST IA-5(1): Authenticator Management - Password-Based Authentication.
#' Validates format, length, and character composition to prevent weak or malformed keys.
#'
#' @examples
#' \dontrun{
#' result <- validate_api_key("sk-proj...")
#' if (result$valid) cat("Provider:", result$provider)
#' }
validate_api_key <- function(api_key, strict = TRUE) {
  if (is.null(api_key) || !is.character(api_key) || !nzchar(api_key)) {
    return(list(valid = FALSE, provider = NULL, error = "API key is NULL, empty, or not a character string"))
  }

  # Detect provider from prefix
  if (grepl("^sk-", api_key)) {
    provider <- "openai"
    min_length <- 40
    length_msg <- "OpenAI keys are typically 48+ characters"
  } else if (grepl("^AIza", api_key)) {
    provider <- "gemini"
    min_length <- 39
    length_msg <- "Gemini keys are typically 39 characters"
  } else {
    return(list(
      valid = FALSE,
      provider = NULL,
      error = "Unknown API key format. OpenAI keys start with 'sk-', Gemini keys start with 'AIza'"
    ))
  }

  if (nchar(api_key) < min_length) {
    return(list(valid = FALSE, provider = provider, error = paste("API key appears too short:", length_msg)))
  }

  if (strict) {
    if (grepl("\\s", api_key)) {
      return(list(valid = FALSE, provider = provider, error = "API key contains whitespace characters"))
    }

    if (grepl("[^A-Za-z0-9_-]", api_key)) {
      return(list(valid = FALSE, provider = provider, error = "API key contains unexpected special characters"))
    }
  }

  return(list(valid = TRUE, provider = provider, error = NULL))
}

#' Validate Column Name
#'
#' @description
#' Validates column names to prevent code injection through formula construction.
#' Ensures column names follow R naming conventions and contain no malicious patterns.
#'
#' @param col_name Character string containing the column name
#'
#' @return TRUE if valid, stops with error if invalid
#' @keywords internal
#'
#' @section Security:
#' Protects against formula injection attacks where malicious column names could
#' execute arbitrary code when used in model formulas. Part of NIST SI-10 input validation.
#'
#' @examples
#' \dontrun{
#' validate_column_name("age")
#' validate_column_name("my_variable")
#' }
validate_column_name <- function(col_name) {
  if (is.null(col_name) || !is.character(col_name) || !nzchar(col_name)) {
    stop("Column name is NULL, empty, or not a character string")
  }

  if (length(col_name) != 1) {
    stop("Column name must be a single value, not a vector")
  }

  if (!grepl("^[A-Za-z][A-Za-z0-9_\\.]*$", col_name)) {
    stop("Invalid column name format. Column names must start with a letter and contain only letters, numbers, underscores, or periods.")
  }

  if (nchar(col_name) > 255) {
    stop("Column name exceeds maximum length of 255 characters")
  }

  backticks <- grepl("`", col_name, fixed = TRUE)
  if (backticks) {
    stop("Column name contains backticks which are not allowed")
  }

  return(TRUE)
}


################################################################################
# WEB ACCESSIBILITY UTILITIES
################################################################################

#' Web Accessibility Utility Functions
#'
#' @description Functions for ensuring WCAG 2.1 Level AA compliance in the Shiny application
#'
#' @section WCAG 2.1 Level AA Compliance:
#' This package follows Web Content Accessibility Guidelines (WCAG) 2.1 Level AA:
#' - 1.1.1 Non-text Content (Level A): Alt text for images and visualizations
#' - 1.4.3 Contrast Minimum (Level AA): 4.5:1 ratio for normal text, 3:1 for large text/UI
#' - 2.1.1 Keyboard (Level A): Full keyboard navigation support
#' - 2.4.1 Bypass Blocks (Level A): Skip navigation links
#' - 3.1.1 Language of Page (Level A): Page language identification
#' - 4.1.2 Name, Role, Value (Level A): ARIA labels and roles

#' Calculate Color Contrast Ratio
#'
#' @description
#' Calculates the contrast ratio between two colors according to WCAG 2.1 standards
#' using the relative luminance formula from W3C guidelines.
#' Used to verify text/background color combinations meet accessibility requirements.
#'
#' @param foreground Foreground color (hex format, e.g., "#111827")
#' @param background Background color (hex format, e.g., "#ffffff")
#'
#' @return Numeric contrast ratio (1-21)
#' @keywords internal
#'
#' @section WCAG Requirements:
#' - Normal text: Minimum 4.5:1 (Level AA)
#' - Large text (18pt+ or 14pt+ bold): Minimum 3:1 (Level AA)
#' - UI components and graphics: Minimum 3:1 (Level AA)
#'
#' @examples
#' \dontrun{
#' calculate_contrast_ratio("#111827", "#ffffff")  # Returns ~16:1 (Pass)
#' calculate_contrast_ratio("#6b7280", "#4a5568")  # Returns ~2.8:1 (Fail)
#' }
calculate_contrast_ratio <- function(foreground, background) {
  hex_to_rgb <- function(hex) {
    hex <- gsub("#", "", hex)
    c(
      strtoi(substr(hex, 1, 2), 16L),
      strtoi(substr(hex, 3, 4), 16L),
      strtoi(substr(hex, 5, 6), 16L)
    ) / 255
  }

  relative_luminance <- function(rgb) {
    rgb <- sapply(rgb, function(val) {
      if (val <= 0.03928) {
        val / 12.92
      } else {
        ((val + 0.055) / 1.055)^2.4
      }
    })
    0.2126 * rgb[1] + 0.7152 * rgb[2] + 0.0722 * rgb[3]
  }

  fg_rgb <- hex_to_rgb(foreground)
  bg_rgb <- hex_to_rgb(background)

  l1 <- relative_luminance(fg_rgb)
  l2 <- relative_luminance(bg_rgb)

  if (l1 > l2) {
    ratio <- (l1 + 0.05) / (l2 + 0.05)
  } else {
    ratio <- (l2 + 0.05) / (l1 + 0.05)
  }

  return(round(ratio, 2))
}

#' Check WCAG Contrast Compliance
#'
#' @description
#' Validates if color combination meets WCAG 2.1 Level AA contrast requirements.
#'
#' @param foreground Foreground color (hex format)
#' @param background Background color (hex format)
#' @param large_text Logical, TRUE if text is large (18pt+ or 14pt+ bold)
#'
#' @return Logical TRUE if compliant, FALSE if not
#' @keywords internal
#'
#' @examples
#' \dontrun{
#' check_wcag_contrast("#111827", "#ffffff")  # TRUE (16:1 ratio)
#' check_wcag_contrast("#6b7280", "#4a5568")  # FALSE (2.8:1 ratio)
#' }
check_wcag_contrast <- function(foreground, background, large_text = FALSE) {
  ratio <- calculate_contrast_ratio(foreground, background)
  min_ratio <- if (large_text) 3.0 else 4.5

  if (ratio >= min_ratio) {
    return(TRUE)
  } else {
    warning(
      "WCAG contrast failure: ", ratio, ":1 ratio (requires ", min_ratio, ":1)\n",
      "  Foreground: ", foreground, "\n",
      "  Background: ", background
    )
    return(FALSE)
  }
}

#' Generate ARIA Label
#'
#' @description
#' Creates accessible ARIA label for UI elements.
#'
#' @param element_type Type of element (e.g., "button", "input", "plot")
#' @param action Action or purpose (e.g., "analyze", "download", "visualize")
#' @param context Additional context (optional)
#'
#' @return Character string with ARIA label
#' @keywords internal
#'
#' @examples
#' \dontrun{
#' create_aria_label("button", "analyze", "readability")
#' # Returns: "Analyze readability button"
#' }
create_aria_label <- function(element_type, action, context = NULL) {
  if (!is.null(context)) {
    label <- paste(tools::toTitleCase(action), context, element_type)
  } else {
    label <- paste(tools::toTitleCase(action), element_type)
  }
  return(label)
}

#' Create Screen Reader Text
#'
#' @description
#' Generates visually hidden text for screen readers (WCAG 4.1.2).
#'
#' @param text Text to be read by screen readers
#'
#' @return HTML span with sr-only class
#' @keywords internal
#'
#' @examples
#' \dontrun{
#' create_sr_text("Loading results, please wait")
#' }
create_sr_text <- function(text) {
  return(
    paste0(
      '<span class="sr-only" role="status" aria-live="polite">',
      text,
      '</span>'
    )
  )
}

#' Validate Keyboard Navigation
#'
#' @description
#' Checks if interactive elements have proper tabindex and keyboard handlers.
#' Used for WCAG 2.1.1 (Keyboard) compliance.
#'
#' @param tabindex Integer, tab order (-1 for no tab, 0 for natural order, 1+ for specific order)
#'
#' @return Logical TRUE if valid, FALSE with warning if invalid
#' @keywords internal
#'
#' @examples
#' \dontrun{
#' validate_keyboard_navigation(0)   # TRUE
#' validate_keyboard_navigation(999) # FALSE (too high)
#' }
validate_keyboard_navigation <- function(tabindex = 0) {
  if (!is.numeric(tabindex)) {
    warning("Tabindex must be numeric")
    return(FALSE)
  }

  if (tabindex > 100) {
    warning("Tabindex > 100 creates unpredictable tab order (WCAG 2.1.1)")
    return(FALSE)
  }

  return(TRUE)
}

#' Check Alt Text Presence
#'
#' @description
#' Validates that images and visualizations have alternative text descriptions.
#' Required for WCAG 1.1.1 (Non-text Content).
#'
#' Note: Decorative images should use empty alt text (alt="") to indicate
#' they should be ignored by assistive technology.
#'
#' @param alt_text Alternative text description
#' @param element_type Type of element (e.g., "plot", "image", "icon")
#' @param decorative Logical, TRUE if element is purely decorative
#'
#' @return Logical TRUE if valid, FALSE with warning if missing/inadequate
#' @keywords internal
#'
#' @examples
#' \dontrun{
#' check_alt_text("Bar chart showing word frequency", "plot")  # TRUE
#' check_alt_text("", "plot")  # FALSE (informative content needs alt text)
#' check_alt_text("", "icon", decorative = TRUE)  # TRUE (decorative is OK)
#' }
check_alt_text <- function(alt_text, element_type = "image", decorative = FALSE) {
  if (decorative) {
    return(TRUE)
  }

  if (is.null(alt_text) || !nzchar(alt_text)) {
    warning("Missing alt text for ", element_type, " (WCAG 1.1.1)")
    return(FALSE)
  }

  if (nchar(alt_text) < 10) {
    warning("Alt text too short for ", element_type, " (consider more descriptive text)")
    return(FALSE)
  }

  return(TRUE)
}


################################################################################
# PLOT HELPER FUNCTIONS
################################################################################

#' Apply Standard Plotly Layout
#'
#' @description
#' Applies consistent layout styling to plotly plots following TextAnalysisR design standards.
#' This ensures all plots have uniform fonts, colors, margins, and interactive features.
#'
#' @param plot A plotly plot object
#' @param title Plot title text (optional)
#' @param xaxis_title X-axis title (optional)
#' @param yaxis_title Y-axis title (optional)
#' @param margin List of margins: list(t, b, l, r) in pixels (default: list(t = 60, b = 80, l = 80, r = 40))
#' @param show_legend Logical, whether to show legend (default: FALSE)
#'
#' @return A plotly plot object with standardized layout
#'
#' @details
#' Design standards applied:
#' - Title: 18px Roboto, #0c1f4a
#' - Axis titles: 16px Roboto, #0c1f4a
#' - Axis tick labels: 16px Roboto, #3B3B3B
#' - Hover tooltips: 16px Roboto
#' - WCAG AA compliant colors
#'
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' library(plotly)
#' p <- plot_ly(x = 1:10, y = rnorm(10), type = "scatter", mode = "markers")
#' p %>% apply_standard_plotly_layout(
#'   title = "My Plot",
#'   xaxis_title = "X Values",
#'   yaxis_title = "Y Values"
#' )
#' }
apply_standard_plotly_layout <- function(plot,
                                         title = NULL,
                                         xaxis_title = NULL,
                                         yaxis_title = NULL,
                                         margin = list(t = 60, b = 80, l = 80, r = 40),
                                         show_legend = FALSE) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  layout_config <- list(
    font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
    hoverlabel = list(
      font = list(size = 16, family = "Roboto, sans-serif"),
      align = "left"
    ),
    margin = margin,
    showlegend = show_legend,
    xaxis = list(
      tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
      titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
    ),
    yaxis = list(
      tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
      titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
    )
  )

  if (!is.null(title)) {
    layout_config$title <- list(
      text = title,
      font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
    )
  }

  if (!is.null(xaxis_title)) {
    layout_config$xaxis$title <- list(
      text = xaxis_title,
      font = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
    )
  }

  if (!is.null(yaxis_title)) {
    layout_config$yaxis$title <- list(
      text = yaxis_title,
      font = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
    )
  }

  do.call(plotly::layout, c(list(p = plot), layout_config)) %>%
    plotly::config(displayModeBar = TRUE)
}


#' Get Standard Plotly Hover Label Configuration
#'
#' @description
#' Returns standardized hover label styling for plotly plots.
#'
#' @param bgcolor Background color (default: "#ffffff")
#' @param fontcolor Font color (default: "#0c1f4a")
#'
#' @return A list of hover label configuration parameters
#'
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' hover_config <- get_plotly_hover_config()
#' plot_ly(..., hoverlabel = hover_config)
#' }
get_plotly_hover_config <- function(bgcolor = "#ffffff", fontcolor = "#0c1f4a") {
  list(
    bgcolor = bgcolor,
    bordercolor = bgcolor,
    font = list(
      family = "Roboto, sans-serif",
      size = 16,
      color = fontcolor
    ),
    align = "left",
    namelength = -1
  )
}


#' Create Standard ggplot2 Theme
#'
#' @description
#' Returns a standardized ggplot2 theme matching TextAnalysisR design standards.
#'
#' @param base_size Base font size (default: 14)
#'
#' @return A ggplot2 theme object
#'
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' library(ggplot2)
#' data(SpecialEduTech, package = "TextAnalysisR")
#' # Create a simple plot using text lengths
#' df <- data.frame(
#'   title_length = nchar(SpecialEduTech$title),
#'   abstract_length = nchar(SpecialEduTech$abstract)
#' )
#' ggplot(df, aes(title_length, abstract_length)) +
#'   geom_point() +
#'   create_standard_ggplot_theme()
#' }
create_standard_ggplot_theme <- function(base_size = 14) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required. Please install it.")
  }

  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        size = 18,
        color = "#0c1f4a",
        hjust = 0.5,
        family = "Roboto"
      ),
      axis.title = ggplot2::element_text(
        size = 16,
        color = "#0c1f4a",
        family = "Roboto"
      ),
      axis.text = ggplot2::element_text(
        size = 16,
        color = "#3B3B3B",
        family = "Roboto"
      ),
      strip.text = ggplot2::element_text(
        size = 16,
        color = "#0c1f4a",
        family = "Roboto"
      ),
      legend.text = ggplot2::element_text(
        size = 16,
        color = "#3B3B3B",
        family = "Roboto"
      ),
      legend.title = ggplot2::element_text(
        size = 16,
        color = "#0c1f4a",
        family = "Roboto"
      )
    )
}


#' Get Sentiment Color Palette
#'
#' @description
#' Returns standardized color mapping for sentiment analysis.
#'
#' @return Named vector of colors
#'
#' @family visualization
#' @export
get_sentiment_colors <- function() {
  c(
    "positive" = "#10B981",
    "negative" = "#EF4444",
    "neutral" = "#6B7280"
  )
}


#' Generate Sentiment Color Gradient
#'
#' @description
#' Generates a color based on sentiment score using a gradient from red (negative)
#' through gray (neutral) to green (positive).
#'
#' @param score Numeric sentiment score (typically -1 to 1)
#'
#' @return Hex color string
#'
#' @family visualization
#' @export
#'
#' @examples
#' get_sentiment_color(-0.8)  # Red
#' get_sentiment_color(0)     # Gray
#' get_sentiment_color(0.8)   # Green
get_sentiment_color <- function(score) {
  normalized_score <- (score + 1) / 2
  normalized_score <- pmax(0, pmin(1, normalized_score))

  if (normalized_score < 0.5) {
    t <- normalized_score * 2
    r <- round(185 * (1 - t) + 75 * t)
    g <- round(67 * (1 - t) + 181 * t)
    b <- round(68 * (1 - t) + 67 * t)
  } else {
    t <- (normalized_score - 0.5) * 2
    r <- round(75 * (1 - t) + 16 * t)
    g <- round(181 * (1 - t) + 185 * t)
    b <- round(67 * (1 - t) + 129 * t)
  }

  sprintf("#%02X%02X%02X", r, g, b)
}


#' Create Message Data Table
#'
#' @description
#' Creates a formatted DT::datatable displaying an informational message.
#' Useful for showing status messages in place of empty tables.
#'
#' @param message Character string message to display
#' @param font_size Font size (default: "16px")
#' @param color Text color (default: "#6c757d")
#'
#' @return A DT::datatable object
#'
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' create_message_table("No data available. Please run analysis first.")
#' }
create_message_table <- function(message,
                                 font_size = "16px",
                                 color = "#6c757d") {

  if (!requireNamespace("DT", quietly = TRUE)) {
    stop("Package 'DT' is required. Please install it.")
  }

  DT::datatable(
    data.frame(Message = message),
    rownames = FALSE,
    options = list(
      dom = "t",
      ordering = FALSE,
      columnDefs = list(
        list(className = 'dt-center', targets = "_all")
      ),
      initComplete = htmlwidgets::JS(
        sprintf(
          "function(settings, json) {
            $(this.api().table().container()).find('td').css({
              'font-size': '%s',
              'color': '%s',
              'padding': '40px',
              'text-align': 'center'
            });
          }",
          font_size,
          color
        )
      )
    ),
    class = 'cell-border stripe'
  )
}


#' Create Empty Plot with Message
#'
#' @description
#' Creates an empty plotly plot displaying a centered message.
#' Useful for showing status messages, error states, or empty data notifications.
#'
#' @param message Character string message to display
#' @param color Text color (default: "#6B7280")
#' @param font_size Font size in pixels (default: 16)
#'
#' @return A plotly object with centered message annotation
#'
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' create_empty_plot_message("No data available")
#' create_empty_plot_message("Click 'Run Analysis' to begin", color = "#337ab7")
#' }
create_empty_plot_message <- function(message,
                                       color = "#6B7280",
                                       font_size = 16) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  plotly::plot_ly(type = "scatter", mode = "markers") %>%
    plotly::layout(
      xaxis = list(
        showgrid = FALSE,
        zeroline = FALSE,
        showticklabels = FALSE,
        title = ""
      ),
      yaxis = list(
        showgrid = FALSE,
        zeroline = FALSE,
        showticklabels = FALSE,
        title = ""
      ),
      annotations = list(
        list(
          text = message,
          x = 0.5,
          y = 0.5,
          xref = "paper",
          yref = "paper",
          showarrow = FALSE,
          font = list(
            size = font_size,
            color = color,
            family = "Roboto, sans-serif"
          )
        )
      )
    ) %>%
    plotly::config(displayModeBar = FALSE)
}


#' Get Standard DataTable Options
#'
#' @description
#' Returns standardized DT::datatable options for consistent table formatting
#' across the TextAnalysisR application.
#'
#' @param scroll_y Vertical scroll height (default: "400px")
#' @param page_length Number of rows per page (default: 25)
#' @param show_buttons Whether to show export buttons (default: TRUE
#'
#' @return A list of DT options
#'
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' DT::datatable(my_data, options = get_dt_options())
#' DT::datatable(my_data, options = get_dt_options(scroll_y = "300px"))
#' }
get_dt_options <- function(scroll_y = "400px",
                            page_length = 25,
                            show_buttons = TRUE) {
  opts <- list(
    scrollX = TRUE,
    scrollY = scroll_y,
    pageLength = page_length
  )

  if (show_buttons) {
    opts$dom <- "Bfrtip"
    opts$buttons <- c("copy", "csv", "excel", "pdf", "print")
  }

  opts
}


################################################################################
# SHINY UI HELPER FUNCTIONS
################################################################################

#' Show Loading/Progress Notification
#'
#' @description
#' Displays a persistent loading notification with a specific ID that can be removed later.
#'
#' @param message The loading message to display
#' @param id Notification ID for later removal (optional)
#'
#' @return Displays a Shiny notification. Returns NULL invisibly.
#'
#' @export
#'
#' @importFrom shiny showNotification
show_loading_notification <- function(message, id = NULL) {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  shiny::showNotification(
    message,
    type = "message",
    duration = NULL,
    id = id
  )

  invisible(NULL)
}

#' Show Completion Notification
#'
#' @description
#' Displays a temporary success notification when a task completes.
#'
#' @param message The completion message to display
#' @param duration Duration in seconds (default: 5)
#'
#' @return Displays a Shiny notification. Returns NULL invisibly.
#'
#' @export
#'
#' @importFrom shiny showNotification
show_completion_notification <- function(message, duration = 5) {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  shiny::showNotification(
    message,
    type = "message",
    duration = duration
  )

  invisible(NULL)
}

#' Show Error Notification
#'
#' @description
#' Displays an error notification to the user.
#'
#' @param message The error message to display
#' @param duration Duration in seconds (default: 7)
#'
#' @return Displays a Shiny notification. Returns NULL invisibly.
#'
#' @export
#'
#' @importFrom shiny showNotification
show_error_notification <- function(message, duration = 7) {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  shiny::showNotification(
    message,
    type = "error",
    duration = duration
  )

  invisible(NULL)
}

#' Show Warning Notification
#'
#' @description
#' Displays a warning notification to the user.
#'
#' @param message The warning message to display
#' @param duration Duration in seconds (default: 5)
#'
#' @return Displays a Shiny notification. Returns NULL invisibly.
#'
#' @export
#'
#' @importFrom shiny showNotification
show_warning_notification <- function(message, duration = 5) {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  shiny::showNotification(
    message,
    type = "warning",
    duration = duration
  )

  invisible(NULL)
}

#' Remove Notification by ID
#'
#' @description
#' Removes a notification with a specific ID.
#'
#' @param id The notification ID to remove
#'
#' @return Removes a Shiny notification. Returns NULL invisibly.
#'
#' @export
#'
#' @importFrom shiny removeNotification
remove_notification_by_id <- function(id) {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  shiny::removeNotification(id)

  invisible(NULL)
}

#' Show No DFM Notification
#'
#' @description
#' Displays a standardized error notification when DFM is required but not available.
#' Shorter alternative to the modal dialog for simple error messages.
#'
#' @param feature_name Name of the feature requiring DFM (default: "this feature")
#' @param duration Duration in seconds (default: 7)
#'
#' @return Displays a Shiny notification. Returns NULL invisibly.
#'
#' @export
#'
#' @importFrom shiny showNotification
show_no_dfm_notification <- function(feature_name = "this feature", duration = 7) {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  message <- paste0(
    "No document-feature matrix available. ",
    "Please complete preprocessing (at least Step 4: DFM) first."
  )

  shiny::showNotification(
    message,
    type = "error",
    duration = duration
  )

  invisible(NULL)
}

#' Show Feature Matrix Notification
#'
#' @description
#' Displays error notification when feature matrix is required but not available.
#' Similar to show_no_dfm_notification but uses "feature matrix" terminology.
#'
#' @param duration Duration in seconds (default: 7)
#'
#' @return Displays a Shiny notification. Returns NULL invisibly.
#'
#' @export
#'
#' @importFrom shiny showNotification
show_no_feature_matrix_notification <- function(duration = 7) {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  shiny::showNotification(
    "No feature matrix available. Please complete preprocessing (at least Step 4: DFM) first.",
    type = "error",
    duration = duration
  )

  invisible(NULL)
}

#' Show Unite Texts Required Notification
#'
#' @description
#' Displays error notification when Step 1 (Unite Texts) is required.
#'
#' @param duration Duration in seconds (default: 5)
#'
#' @return Displays a Shiny notification. Returns NULL invisibly.
#'
#' @export
#'
#' @importFrom shiny showNotification
show_unite_texts_required_notification <- function(duration = 5) {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  shiny::showNotification(
    "Please create united texts first in the preprocessing steps (Step 1: Unite texts).",
    type = "error",
    duration = duration
  )

  invisible(NULL)
}

#' Show Guide Modal Dialog from HTML File
#'
#' @description
#' Loads and displays a modal dialog with guide content from an HTML file.
#' This function is designed for Shiny applications to display help documentation
#' stored in external HTML files, reducing server.R file size and improving
#' maintainability.
#'
#' @param guide_name Name of the guide file (without .html extension).
#'   Files should be located in inst/TextAnalysisR.app/markdown/guides/
#' @param title Modal dialog title to display
#' @param size Size of the modal dialog (default: "l" for large).
#'   Options: "s" (small), "m" (medium), "l" (large)
#'
#' @return Displays a Shiny modal dialog. Returns NULL invisibly.
#'
#' @details
#' Guide HTML files should be placed in:
#' \code{inst/TextAnalysisR.app/markdown/guides/<guide_name>.html}
#'
#' The function will look for the guide file in the installed package location.
#' If the file is not found, it displays an error message in the modal.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' observeEvent(input$showDimRedInfo, {
#'   show_guide_modal("dimensionality_reduction_guide", "Dimensionality Reduction Guide")
#' })
#'
#' observeEvent(input$showClusteringInfo, {
#'   show_guide_modal("clustering_guide", "Document Clustering Guide")
#' })
#' }
#'
#' @importFrom shiny showModal modalDialog modalButton
show_guide_modal <- function(guide_name, title, size = "l") {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  if (!requireNamespace("htmltools", quietly = TRUE)) {
    stop("The 'htmltools' package is required for this function.")
  }

  guide_path <- system.file(
    "TextAnalysisR.app", "markdown", "guides",
    paste0(guide_name, ".html"),
    package = "TextAnalysisR"
  )

  if (!file.exists(guide_path) || guide_path == "") {
    guide_path <- file.path(
      "inst", "TextAnalysisR.app", "markdown", "guides",
      paste0(guide_name, ".html")
    )
  }

  if (!file.exists(guide_path)) {
    guide_path <- file.path(
      "markdown", "guides",
      paste0(guide_name, ".html")
    )
  }

  if (file.exists(guide_path)) {
    content <- htmltools::HTML(paste(readLines(guide_path, warn = FALSE), collapse = "\n"))
  } else {
    content <- htmltools::tags$p(
      paste0("Guide content not found: ", guide_name, ".html"),
      style = "color: #DC2626;"
    )
  }

  shiny::showModal(
    shiny::modalDialog(
      title = title,
      size = size,
      content,
      footer = shiny::modalButton("Close"),
      easyClose = TRUE
    )
  )

  invisible(NULL)
}

#' Show DFM Requirement Modal
#'
#' @description
#' Displays a standardized modal dialog informing users they need to complete
#' preprocessing steps before using a feature that requires a document-feature matrix.
#'
#' @param feature_name Name of the feature requiring DFM (e.g., "topic modeling", "keyword extraction")
#' @param additional_message Optional additional message to display (default: NULL)
#'
#' @return Displays a Shiny modal dialog. Returns NULL invisibly.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' if (is.null(dfm_init())) {
#'   show_dfm_required_modal("topic modeling")
#'   return(NULL)
#' }
#' }
#'
#' @importFrom shiny showModal modalDialog modalButton tags p
show_dfm_required_modal <- function(feature_name = "this feature", additional_message = NULL) {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  message_content <- list(
    shiny::p("No document-feature matrix (DFM) found."),
    shiny::p("Please complete the required preprocessing steps:")
  )

  if (!is.null(additional_message)) {
    message_content <- c(message_content, list(shiny::p(additional_message)))
  }

  message_content <- c(
    message_content,
    list(
      shiny::tags$div(
        style = "margin-left: 20px; margin-top: 10px;",
        shiny::tags$p(
          shiny::tags$strong(style = "color: #DC2626;", "Required:"),
          style = "margin-bottom: 5px;"
        ),
        shiny::tags$ul(
          shiny::tags$li(shiny::tags$strong("Step 1:"), " Unite Texts"),
          shiny::tags$li(shiny::tags$strong("Step 4:"), " Document-Feature Matrix (DFM)")
        ),
        shiny::tags$p(
          shiny::tags$strong(style = "color: #6B7280;", "Optional:"),
          " Steps 2, 3, 5, and 6",
          style = "margin-top: 10px; font-size: 12px;"
        )
      )
    )
  )

  shiny::showModal(
    shiny::modalDialog(
      title = "Preprocessing Required",
      message_content,
      easyClose = TRUE,
      footer = shiny::modalButton("OK")
    )
  )

  invisible(NULL)
}

#' Show Preprocessing Steps Modal
#'
#' @description
#' Displays a modal dialog listing required preprocessing steps for a feature.
#' Generic version that works for any feature requiring preprocessing.
#'
#' @param title Modal title (default: "Preprocessing Required")
#' @param message Main message to display
#' @param required_steps Character vector of required preprocessing steps
#' @param optional_steps Character vector of optional preprocessing steps (default: NULL)
#' @param additional_note Optional additional note to display (default: NULL)
#'
#' @return Displays a Shiny modal dialog. Returns NULL invisibly.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' show_preprocessing_steps_modal(
#'   message = "Please complete preprocessing to generate tokens.",
#'   required_steps = c("Step 1: Unite Texts", "Step 4: Document-Feature Matrix"),
#'   optional_steps = c("Steps 2, 3, 5, and 6")
#' )
#' }
#'
#' @importFrom shiny showModal modalDialog modalButton tags p
show_preprocessing_steps_modal <- function(title = "Preprocessing Required",
                                          message,
                                          required_steps,
                                          optional_steps = NULL,
                                          additional_note = NULL) {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  content <- list(shiny::p(message))

  steps_div <- shiny::tags$div(
    style = "margin-left: 20px; margin-top: 10px;",
    shiny::tags$p(
      shiny::tags$strong(style = "color: #DC2626;", "Required:"),
      style = "margin-bottom: 5px;"
    ),
    shiny::tags$ul(
      lapply(required_steps, function(step) shiny::tags$li(step))
    )
  )

  if (!is.null(optional_steps)) {
    steps_div <- shiny::tagAppendChild(
      steps_div,
      shiny::tags$p(
        shiny::tags$strong(style = "color: #6B7280;", "Optional:"),
        paste(optional_steps, collapse = ", "),
        style = "margin-top: 10px; font-size: 12px;"
      )
    )
  }

  content <- c(content, list(steps_div))

  if (!is.null(additional_note)) {
    content <- c(content, list(shiny::p(additional_note, style = "margin-top: 10px; font-size: 12px; color: #6B7280;")))
  }

  shiny::showModal(
    shiny::modalDialog(
      title = title,
      content,
      easyClose = TRUE,
      footer = shiny::modalButton("OK")
    )
  )

  invisible(NULL)
}

#' Generate DFM Setup Instructions Text
#'
#' @description
#' Generates standardized text instructions for creating a DFM.
#' Used in console output or verbatim text displays.
#'
#' @param feature_name Name of the feature requiring DFM (default: "this feature")
#'
#' @return Character vector of instruction lines
#'
#' @export
#'
#' @examples
#' \dontrun{
#' output$instructions <- renderPrint({
#'   cat(get_dfm_setup_instructions("keyword extraction"), sep = "\n")
#' })
#' }
get_dfm_setup_instructions <- function(feature_name = "this feature") {
  c(
    "Warning: DFM Processing Required\n",
    "Please complete the following steps first:\n",
    "1. Go to the 'Preprocess' tab",
    "2. Navigate to Step 4: Document-Feature Matrix",
    "3. Click the 'Process' button\n",
    paste0("Once the DFM is created, you can return here to use ", feature_name, ".")
  )
}

#' Show DFM Setup Instructions Modal
#'
#' @description
#' Displays a modal dialog with console-style instructions for creating a DFM.
#' Uses verbatimTextOutput for formatting.
#'
#' @param output_id Shiny output ID for the verbatimTextOutput
#' @param feature_name Name of the feature requiring DFM (default: "this feature")
#' @param session Shiny session object (default: getDefaultReactiveDomain())
#'
#' @return Displays a Shiny modal dialog. Returns NULL invisibly.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' output$dfm_instructions <- renderPrint({
#'   cat(get_dfm_setup_instructions("keywords"), sep = "\n")
#' })
#'
#' show_dfm_instructions_modal("dfm_instructions", "keywords")
#' }
#'
#' @importFrom shiny showModal modalDialog modalButton verbatimTextOutput getDefaultReactiveDomain
show_dfm_instructions_modal <- function(output_id, feature_name = "this feature", session = NULL) {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  if (is.null(session)) {
    session <- shiny::getDefaultReactiveDomain()
  }

  shiny::showModal(
    shiny::modalDialog(
      title = "DFM Required",
      shiny::verbatimTextOutput(output_id),
      easyClose = TRUE,
      footer = shiny::modalButton("Close")
    )
  )

  invisible(NULL)
}

#' Show Generic Preprocessing Required Modal
#'
#' @description
#' Displays a simple modal indicating preprocessing is required.
#' Lightweight alternative when detailed steps aren't needed.
#'
#' @param message Custom message (default: "Please complete preprocessing steps first.")
#' @param title Modal title (default: "Preprocessing Required")
#'
#' @return Displays a Shiny modal dialog. Returns NULL invisibly.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' if (!preprocessing_complete()) {
#'   show_preprocessing_required_modal()
#'   return()
#' }
#' }
#'
#' @importFrom shiny showModal modalDialog modalButton p
show_preprocessing_required_modal <- function(message = "Please complete preprocessing steps first.",
                                             title = "Preprocessing Required") {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("The 'shiny' package is required for this function.")
  }

  shiny::showModal(
    shiny::modalDialog(
      title = title,
      shiny::p(message),
      easyClose = TRUE,
      footer = shiny::modalButton("OK")
    )
  )

  invisible(NULL)
}


# Text Formatting Utilities ----

#' Truncate Text with Ellipsis
#'
#' @description Truncates text to a maximum number of characters and adds
#'   ellipsis if truncated.
#'
#' @param text Character string to truncate.
#' @param max_chars Maximum number of characters (default: 50).
#'
#' @return Truncated text with "..." appended if truncated.
#'
#' @family text-utilities
#' @export
truncate_text_with_ellipsis <- function(text, max_chars = 50) {
  text <- as.character(text)
  if (nchar(text) <= max_chars) {
    return(text)
  }
  paste0(substr(text, 1, max_chars), "...")
}

#' Truncate Text to Word Count
#'
#' @description Truncates text to a maximum number of words and adds
#'   ellipsis if truncated.
#'
#' @param text Character string to truncate.
#' @param max_words Maximum number of words (default: 150).
#'
#' @return Truncated text with "..." appended if truncated.
#'
#' @family text-utilities
#' @export
truncate_text_to_words <- function(text, max_words = 150) {
  text <- as.character(text)
  words <- strsplit(text, "\\s+")[[1]]

  if (length(words) > max_words) {
    truncated_text <- paste(words[1:max_words], collapse = " ")
    return(paste0(truncated_text, "..."))
  } else {
    return(text)
  }
}

#' Wrap Long Text with Line Breaks
#'
#' @description Wraps long text by inserting line breaks at word boundaries.
#'   Handles both spaced text and continuous text (like URLs).
#'
#' @param text Character string to wrap.
#' @param chars_per_line Maximum characters per line (default: 50).
#' @param max_lines Maximum number of lines (default: 3).
#'
#' @return Text with line breaks inserted.
#'
#' @family text-utilities
#' @export
wrap_long_text <- function(text, chars_per_line = 50, max_lines = 3) {
  text <- as.character(text)

  if (nchar(text) <= chars_per_line) return(text)

  # Handle text without spaces (like URLs)
  if (!grepl(" ", text) && nchar(text) > chars_per_line) {
    lines <- character()
    remaining_text <- text
    while (nchar(remaining_text) > chars_per_line && length(lines) < max_lines - 1) {
      lines <- c(lines, substr(remaining_text, 1, chars_per_line))
      remaining_text <- substr(remaining_text, chars_per_line + 1, nchar(remaining_text))
    }
    if (nchar(remaining_text) > 0) {
      if (nchar(remaining_text) > chars_per_line) {
        lines <- c(lines, paste0(substr(remaining_text, 1, chars_per_line - 3), "..."))
      } else {
        lines <- c(lines, remaining_text)
      }
    }
    return(paste(lines, collapse = "\n"))
  }

  # Handle normal text with spaces
  words <- strsplit(text, " ")[[1]]
  lines <- character()
  current_line <- ""

  for (word in words) {
    if (length(lines) >= max_lines - 1 && current_line != "") {
      lines <- c(lines, paste0(current_line, "..."))
      break
    }

    if (current_line == "") {
      test_line <- word
    } else {
      test_line <- paste(current_line, word)
    }

    if (nchar(test_line) <= chars_per_line) {
      current_line <- test_line
    } else {
      if (current_line != "") {
        lines <- c(lines, current_line)
        current_line <- word
      } else {
        if (length(lines) < max_lines - 1) {
          lines <- c(lines, paste0(substr(word, 1, chars_per_line - 3), "..."))
          current_line <- ""
        }
      }
    }
  }

  if (nchar(current_line) > 0) {
    if (length(lines) < max_lines) {
      lines <- c(lines, current_line)
    }
  }

  paste(lines[1:min(length(lines), max_lines)], collapse = "\n")
}

#' Wrap Text for Tooltip Display
#'
#' @description Formats text for tooltip display with size limits
#'   and line wrapping.
#'
#' @param text Character string to format.
#' @param max_words Maximum words (not currently used, kept for compatibility).
#' @param chars_per_line Maximum characters per line (default: 50).
#' @param max_lines Maximum number of lines (default: 3).
#'
#' @return Formatted text suitable for tooltip display.
#'
#' @family text-utilities
#' @export
wrap_text_for_tooltip <- function(text, max_words = 150, chars_per_line = 50, max_lines = 3) {
  text <- as.character(text)

  # Limit to 150 characters first
  if (nchar(text) > 150) {
    text_to_use <- substr(text, 1, 150)
    needs_ellipsis <- TRUE
  } else {
    text_to_use <- text
    needs_ellipsis <- FALSE
  }

  if (nchar(text_to_use) <= chars_per_line) {
    return(text_to_use)
  }

  result <- ""
  lines_created <- 0
  current_pos <- 1

  while (current_pos <= nchar(text_to_use) && lines_created < max_lines) {
    end_pos <- min(current_pos + chars_per_line - 1, nchar(text_to_use))
    line_text <- substr(text_to_use, current_pos, end_pos)

    if (end_pos < nchar(text_to_use) && lines_created < max_lines - 1) {
      last_space <- regexpr(" [^ ]*$", line_text)
      if (last_space > 0) {
        line_text <- substr(line_text, 1, last_space - 1)
        end_pos <- current_pos + last_space - 2
      }
    }

    if (result == "") {
      result <- line_text
    } else {
      result <- paste0(result, "\n", line_text)
    }

    lines_created <- lines_created + 1
    current_pos <- end_pos + 2
  }

  if (needs_ellipsis || current_pos <= nchar(text_to_use)) {
    result <- paste0(result, "...")
  }

  return(result)
}


# Matrix Utilities ----

#' Clean Similarity Matrix
#'
#' @description Cleans a similarity matrix by handling NA/Inf values,
#'   ensuring symmetry, and setting diagonal to 1.
#'
#' @param similarity_matrix A numeric matrix of similarity values.
#'
#' @return Cleaned similarity matrix.
#'
#' @family matrix-utilities
#' @export
clean_similarity_matrix <- function(similarity_matrix) {
  # Replace non-finite values with 0
  similarity_matrix[!is.finite(similarity_matrix)] <- 0

  # Ensure symmetry
  if (nrow(similarity_matrix) == ncol(similarity_matrix)) {
    similarity_matrix <- (similarity_matrix + t(similarity_matrix)) / 2
  }

  # Set diagonal to 1
  if (nrow(similarity_matrix) == ncol(similarity_matrix)) {
    diag(similarity_matrix) <- 1
  }

  return(similarity_matrix)
}

#' Renumber Clusters Sequentially
#'
#' @description Renumbers cluster assignments to sequential integers
#'   starting from 1.
#'
#' @param clusters A vector of cluster assignments.
#'
#' @return Vector with clusters renumbered sequentially (1, 2, 3, ...).
#'
#' @family matrix-utilities
#' @export
renumber_clusters_sequentially <- function(clusters) {
  if (is.null(clusters) || length(clusters) == 0) {
    return(clusters)
  }

  unique_clusters <- sort(unique(clusters))
  cluster_mapping <- stats::setNames(seq_along(unique_clusters), unique_clusters)
  return(cluster_mapping[as.character(clusters)])
}
