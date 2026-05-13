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
#' @param dev Logical. If `TRUE`, runs an accessibility palette audit
#'   (via `a11yviz::a11y_check_palette()`) on inline hex colors found in the
#'   Shiny source files and prints any pairs that fail WCAG 2.1 AA contrast
#'   against the app's light (`#ffffff`) and dark (`#0d1117`) backgrounds.
#'   Requires the `a11yviz` package.
#'
#' @return No return value, called for side effects (launching Shiny app).
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
#' @import shiny
#' @import tidyr
#' @importFrom magrittr %>%

run_app <- function(launch.browser = interactive(), dev = FALSE) {
  appDir <- system.file("TextAnalysisR.app", package = "TextAnalysisR")

  if (appDir == "") {
    stop("Error: TextAnalysisR.app directory not found.")
  }

  if (isTRUE(dev)) {
    .audit_app_palette(appDir)
  }

  shiny::runApp(appDir, display.mode = "normal", launch.browser = launch.browser)
}

.audit_app_palette <- function(appDir, level = "AA-large", lum_skip = 0.85) {
  if (!requireNamespace("a11yviz", quietly = TRUE)) {
    message("Install 'a11yviz' to enable dev palette audit: ",
            "remotes::install_github('mshin77/a11yviz')")
    return(invisible(NULL))
  }
  files <- list.files(appDir, pattern = "\\.R$", full.names = TRUE)
  txt <- unlist(lapply(files, readLines, warn = FALSE))
  hex <- unique(toupper(stats::na.omit(regmatches(txt, regexpr("#[0-9a-fA-F]{6}", txt)))))
  if (length(hex) == 0) {
    message("Palette audit: no inline hex colors found.")
    return(invisible(NULL))
  }
  # Exclude pale surface colors (likely panel/background, not plot markers).
  rgb_to_lum <- function(h) {
    v <- as.numeric(grDevices::col2rgb(h)) / 255
    v <- ifelse(v <= 0.03928, v / 12.92, ((v + 0.055) / 1.055) ^ 2.4)
    sum(v * c(0.2126, 0.7152, 0.0722))
  }
  lum <- vapply(hex, rgb_to_lum, numeric(1))
  hex_fg <- hex[lum <= lum_skip]
  # Brand/status fill colors paired with dark text on top - not foreground-on-white.
  brand_skip <- c(
    "#CBD5E1", "#F59E0B", "#10B981", "#4CAF50", "#FF9800", "#00BCD4",
    "#94A3B8", "#FF8F00", "#DEE2E6", "#8BC34A", "#FFC107", "#00ACC1",
    "#FEE2E2", "#9CA3AF", "#E0E7FF", "#E5E7EB", "#FFCCCC", "#0EA5E9",
    "#9E9D24", "#E2E8F0"
  )
  hex_fg <- setdiff(hex_fg, brand_skip)
  res <- a11yviz::a11y_check_palette(hex_fg, bg = c("#FFFFFF", "#0D1117"), level = level)
  failing <- res[res$status != "ok", , drop = FALSE]
  if (nrow(failing) == 0) {
    message(sprintf("Palette audit (%s, %d foreground colors): all pass.",
                    level, length(hex_fg)))
  } else {
    message(sprintf("Palette audit (%s): %d/%d pairs fail. Skipped %d pale surfaces.",
                    level, nrow(failing), nrow(res), length(hex) - length(hex_fg)))
    print(failing)
  }
  invisible(res)
}
