#' @title Themes From Coded Units
#'
#' @description
#' Groups codes under the themes named in the codebook and counts the coded
#' units and documents each theme covers. Codes without a theme fall under
#' "(no theme)". Themes are the researcher's grouping; nothing is inferred.
#'
#' @param assignments Table with `unit_id`, `doc_id`, `code`, and `status`, such
#'   as `project$assignments`. Rows with status "human" or "ai-confirmed" count.
#' @param codebook Table with `code` and `theme`, such as `project$codebook`.
#'
#' @return A tibble with `theme`, `codes`, `n_codes`, `n_units`, and `n_docs`.
#'
#' @seealso [code_saturation()] for the code accumulation curve.
#' @concept qualitative-coding
#' @export
code_themes <- function(assignments, codebook) {
  if (!is.data.frame(codebook) || !all(c("code", "theme") %in% names(codebook))) {
    stop("codebook must have columns: code, theme.", call. = FALSE)
  }
  if (!is.data.frame(assignments) || !all(c("unit_id", "doc_id", "code", "status") %in% names(assignments))) {
    stop("assignments must have columns: unit_id, doc_id, code, status.", call. = FALSE)
  }
  theme <- trimws(as.character(codebook$theme))
  theme[is.na(theme) | !nzchar(theme)] <- "(no theme)"
  cb <- tibble::tibble(code = as.character(codebook$code), theme = theme)
  a <- assignments[assignments$status %in% c("human", "ai-confirmed") & !is.na(assignments$code), , drop = FALSE]
  used <- dplyr::inner_join(tibble::tibble(unit_id = a$unit_id, doc_id = a$doc_id, code = as.character(a$code)), cb, by = "code")
  covered <- dplyr::summarise(dplyr::group_by(used, theme), n_units = dplyr::n_distinct(unit_id), n_docs = dplyr::n_distinct(doc_id))
  out <- dplyr::summarise(dplyr::group_by(cb, theme), codes = paste(code, collapse = ", "), n_codes = dplyr::n())
  out <- dplyr::left_join(out, covered, by = "theme")
  out$n_units[is.na(out$n_units)] <- 0L
  out$n_docs[is.na(out$n_docs)] <- 0L
  return(out[order(out$theme == "(no theme)", out$theme), , drop = FALSE])
}
