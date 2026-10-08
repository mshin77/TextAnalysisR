#' @title Launch the TextAnalysisR app
#'
#' @name run_app
#'
#' @description
#' Launch the TextAnalysisR Shiny application.
#'
#' @details
#' The Qualitative Coding tab autosaves its project file
#' (`coding-project-<coder>.rds`) to `tools::R_user_dir("TextAnalysisR", "data")`.
#' Set
#' `options(TextAnalysisR.project_dir = "<path>")` before launching to keep the
#' file next to an analysis project instead. The hosted app does not autosave to disk; there
#' the project is kept on the device and downloaded with "Save project".
#'
#' API keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`) in a `.env` file in the
#' working directory are read for the app session only; variables already set
#' in the R session take precedence.
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

  # trusted: the .env is the user's own file; values last for the app session only
  dotenv <- if (file.exists(".env")) grep("^[A-Za-z_][A-Za-z0-9_]*=", trimws(readLines(".env", warn = FALSE)), value = TRUE) else character(0)
  dotenv <- stats::setNames(gsub("^[\"']|[\"']$", "", sub("^[^=]*=", "", dotenv)), sub("=.*", "", dotenv))
  dotenv <- dotenv[!nzchar(Sys.getenv(names(dotenv)))]
  if (length(dotenv) > 0) withr::local_envvar(dotenv)

  shiny::runApp(appDir, display.mode = "normal", launch.browser = launch.browser)
}
