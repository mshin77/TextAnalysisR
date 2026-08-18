#' @keywords internal
.memo_columns <- function() {
  return(tibble::tibble(
    memo_id = character(0),
    target_type = character(0),
    target_id = character(0),
    round = integer(0),
    text = character(0),
    created = as.POSIXct(character(0))))
}

#' @title Attach an Analytic Memo to a Unit or Category
#'
#' @description
#' Records the reasoning behind a coding decision so the interpretive step leaves
#' a trail. Memos attach to a single unit or to a category, carry the round they
#' were written in, and travel with the coded units on export.
#'
#' @param memos Existing memo table, or `NULL` to start one.
#' @param target_type "unit" or "category".
#' @param target_id Identifier of the unit or category the memo describes.
#' @param text The memo. Free text; nothing parses it.
#' @param round Refinement round the memo belongs to (default 1).
#'
#' @return The memo table with one row appended.
#'
#' @seealso [get_memos()] to retrieve them; [round_summary()] for the rounds
#'   memos are stamped with.
#' @concept qualitative-coding
#' @export
add_memo <- function(memos = NULL, target_type = c("unit", "category"),
                     target_id, text, round = 1) {
  target_type <- match.arg(target_type)
  if (is.null(memos)) memos <- .memo_columns()
  if (!is.character(text) || length(text) != 1 || is.na(text)) {
    stop("text must be a single string.", call. = FALSE)
  }
  if (!nzchar(trimws(text))) {
    stop("text must not be empty.", call. = FALSE)
  }
  if (length(target_id) != 1 || is.na(target_id)) {
    stop("target_id must be a single non-missing value.", call. = FALSE)
  }

  row <- tibble::tibble(
    memo_id = paste0("m", nrow(memos) + 1L),
    target_type = target_type,
    target_id = as.character(target_id),
    round = as.integer(round),
    text = trimws(text),
    created = Sys.time())
  return(dplyr::bind_rows(memos, row))
}

#' @title Retrieve Analytic Memos
#'
#' @description
#' Returns memos filtered by target and round. Omitted filters match everything,
#' so calling with no filter returns the full table in the order written.
#'
#' @param memos Memo table from [add_memo()].
#' @param target_type "unit", "category", or `NULL` for both.
#' @param target_id Identifier to match, or `NULL` for all.
#' @param round Round to match, or `NULL` for all.
#'
#' @return A tibble of matching memos, oldest first.
#'
#' @seealso [add_memo()] to write them.
#' @concept qualitative-coding
#' @export
get_memos <- function(memos, target_type = NULL, target_id = NULL, round = NULL) {
  if (is.null(memos) || nrow(memos) == 0) return(.memo_columns())
  keep <- rep(TRUE, nrow(memos))
  if (!is.null(target_type)) keep <- keep & memos$target_type == target_type
  if (!is.null(target_id)) keep <- keep & memos$target_id == as.character(target_id)
  if (!is.null(round)) keep <- keep & memos$round == as.integer(round)
  return(memos[keep, , drop = FALSE])
}
