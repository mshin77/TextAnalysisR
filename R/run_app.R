#' @title Launch the TextAnalysisR app
#'
#' @name run_app
#'
#' @description
#' Launch the TextAnalysisR Shiny application.
#'
#' @details
#' The Qualitative Coding tab autosaves its project file
#' (`coding-project-<coder>.rds`) to the folder that is the working directory
#' when `run_app()` is called, so start the app from the analysis project's
#' folder. Set `options(TextAnalysisR.project_dir = "<path>")` before
#' launching to save elsewhere. The hosted app does not autosave to disk; there
#' the project is kept on the device and downloaded with "Save project".
#'
#' @param launch.browser Logical. Whether to open the app in a browser.
#'   Defaults to `interactive()`, which is FALSE in non-interactive sessions
#'   (e.g., Docker containers, servers).
#'
#' @return No return value, called for side effects (launching Shiny app).
#'
#' @examples
#' if (interactive()) {
#'   run_app()
#' }
#'
#' @export
#'
#' @import dplyr
#' @import ggplot2
#' @import shiny
#' @import tidyr
#' @importFrom magrittr %>%

run_app <- function(launch.browser = interactive()) {
  appDir <- system.file("TextAnalysisR.app", package = "TextAnalysisR")

  if (appDir == "") {
    stop("Error: TextAnalysisR.app directory not found.")
  }

  app_pkgs <- c("shinyjs", "shinybusy", "shinyBS", "stringr", "plotly", "markdown", "later", "digest")
  missing_pkgs <- app_pkgs[!vapply(app_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0) {
    stop(sprintf(
      "The app needs these packages: %s\nInstall with: install.packages(c(%s))",
      paste(missing_pkgs, collapse = ", "),
      paste0('"', missing_pkgs, '"', collapse = ", ")
    ), call. = FALSE)
  }

  # global.R sets these when sourced; restore the caller's values on exit
  app_options <- c("shiny.maxRequestSize", "shiny.timeout", "shiny.useragg", "DT.options")
  withr::local_options(lapply(stats::setNames(nm = app_options), getOption))
  withr::local_envvar(Sys.getenv(c("HF_HUB_ETAG_TIMEOUT", "HF_HUB_DISABLE_TELEMETRY"), unset = NA))
  if (is.null(getOption("TextAnalysisR.project_dir"))) withr::local_options(TextAnalysisR.project_dir = getwd())

  shiny::runApp(appDir, display.mode = "normal", launch.browser = launch.browser)
}
