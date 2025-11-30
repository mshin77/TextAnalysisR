#' @importFrom utils modifyList
#' @importFrom stats cor
NULL

# Utility and Helper Functions
# General-purpose utility functions for analysis and visualization

#
# Deployment Detection Utilities
#

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
 shinyapps <- nzchar(Sys.getenv("SHINY_PORT")) ||
    Sys.getenv("R_CONFIG_ACTIVE") == "shinyapps"
  connect <- nzchar(Sys.getenv("RSTUDIO_CONNECT_HASTE"))
  server <- nzchar(Sys.getenv("SHINY_SERVER_VERSION"))
  return(shinyapps || connect || server)
}

#' Check Feature Status
#'
#' @description
#' Checks if a specific optional feature is available in the current environment.
#'
#' @param feature Character: "python", "ollama", "langgraph", "pdf_tables", "embeddings"
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
    "langgraph" = tryCatch({
      status <- check_python_env()
      isTRUE(status$available) && isTRUE(status$packages$langgraph)
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
#' status <- get_feature_status()
#' print(status)
get_feature_status <- function() {
  features <- c("python", "ollama", "langgraph", "pdf_tables", "embeddings", "sentiment_deep")
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
    disabled <- c("Python PDF processing", "Ollama/LangGraph AI",
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
    "langgraph" = "LangGraph not available. Run setup_python_env().",
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
        tickfont = list(size = 16, color = "#3B3B3B"),
        titlefont = list(size = 16, color = "#0c1f4a")
      ),
      yaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B"),
        titlefont = list(size = 16, color = "#0c1f4a")
      ),
      hoverlabel = list(
        font = list(size = 15)
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
