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
