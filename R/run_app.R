#' @title Launch the TextAnalysisR app
#'
#' @name run_app
#'
#' @description
#' Launch the TextAnalysisR Shiny application.
#'
#' @param launch.browser Logical. Whether to open the app in a browser.
#'   Defaults to `interactive()`, which is FALSE in non-interactive sessions
#'   (e.g., Docker containers, servers).
#'
#' @return No return value, called for side effects (launching Shiny app)
#'
#' @examples
#' if (interactive()) {
#'   library(TextAnalysisR)
#'   run_app()
#' }
#'
#' @export
#'
#' @import dplyr
#' @import ggplot2
#' @import ggraph
#' @import shiny
#' @import stm
#' @import tidyr
#' @import tidytext
#' @import widyr
#' @importFrom magrittr %>%

run_app <- function(launch.browser = interactive()) {
  appDir <- system.file("TextAnalysisR.app", package = "TextAnalysisR")

  if (appDir == "") {
    stop("Error: TextAnalysisR.app directory not found.")
  }

  # Check Python environment on first run
  python_check <- tryCatch({
    check_python_env()
  }, error = function(e) {
    list(available = FALSE)
  })

  if (!python_check$available) {
    message("\nOptional: For PDF tables and AI features, run setup_python_env()")
    response <- readline(prompt = "Run setup now? (y/n): ")

    if (tolower(trimws(response)) == "y") {
      tryCatch({
        setup_python_env()
        message("Setup complete. Restart R and relaunch app.")
      }, error = function(e) {
        message("Setup failed: ", e$message)
      })
    }
  }

  shiny::runApp(appDir, display.mode = "normal", launch.browser = launch.browser)
}
