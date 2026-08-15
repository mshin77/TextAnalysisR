#' @keywords internal
.validate_codebook <- function(codebook) {
  if (!is.data.frame(codebook) || nrow(codebook) == 0) {
    stop("codebook must be a non-empty data frame.", call. = FALSE)
  }
  need <- c("code", "definition")
  miss <- setdiff(need, names(codebook))
  if (length(miss) > 0) {
    stop("codebook is missing required column(s): ", paste(miss, collapse = ", "), call. = FALSE)
  }
  codebook$code <- as.character(codebook$code)
  if (anyDuplicated(codebook$code) > 0) {
    stop("codebook has duplicate codes.", call. = FALSE)
  }
  codebook$definition <- as.character(codebook$definition)
  example <- if ("example" %in% names(codebook)) as.character(codebook$example) else NA_character_
  color <- if ("color" %in% names(codebook)) as.character(codebook$color) else NA_character_
  tibble::tibble(
    code = codebook$code,
    definition = codebook$definition,
    example = example,
    color = color
  )
}

#' @keywords internal
.split_units <- function(text, unit) {
  empty <- tibble::tibble(unit_text = character(0), start = integer(0), end = integer(0))
  if (is.na(text) || !nzchar(trimws(text))) return(empty)
  if (unit == "document") {
    return(tibble::tibble(unit_text = text, start = 1L, end = nchar(text)))
  }
  pattern <- if (unit == "paragraph") "\\n[ \\t]*\\n+" else "(?<=[.!?])\\s+"
  parts <- trimws(strsplit(text, pattern, perl = TRUE)[[1]])
  parts <- parts[nzchar(parts)]
  if (length(parts) == 0) return(empty)
  pos <- 1L
  bounds <- vapply(parts, function(p) {
    hit <- regexpr(p, substr(text, pos, nchar(text)), fixed = TRUE)
    offset <- if (hit < 0) 0L else as.integer(hit) - 1L
    start <- pos + offset
    pos <<- start + nchar(p)
    c(start, start + nchar(p) - 1L)
  }, integer(2), USE.NAMES = FALSE)
  tibble::tibble(unit_text = parts, start = bounds[1, ], end = bounds[2, ])
}

#' @keywords internal
.parse_codes_response <- function(response, valid_codes, max_codes) {
  none <- tibble::tibble(code = NA_character_, confidence = NA_real_, rationale = NA_character_)
  start <- regexpr("\\{", response)
  end <- regexpr("\\}(?=[^}]*$)", response, perl = TRUE)
  if (start < 1 || end < 1 || end < start) return(none)
  parsed <- tryCatch(
    jsonlite::fromJSON(substr(response, start, end), simplifyVector = TRUE),
    error = function(e) NULL
  )
  a <- tryCatch(tibble::as_tibble(parsed$assignments), error = function(e) NULL)
  if (is.null(a) || nrow(a) == 0 || !"code" %in% names(a)) return(none)
  a$code <- as.character(a$code)
  a <- a[!is.na(a$code) & a$code %in% valid_codes & !duplicated(a$code), , drop = FALSE]
  if (nrow(a) == 0) return(none)
  a <- a[seq_len(min(nrow(a), max_codes)), , drop = FALSE]
  conf <- if ("confidence" %in% names(a)) suppressWarnings(as.numeric(a$confidence)) else rep(NA_real_, nrow(a))
  tibble::tibble(
    code = a$code,
    confidence = pmin(1, pmax(0, conf)),
    rationale = if ("rationale" %in% names(a)) as.character(a$rationale) else NA_character_
  )
}

#' @keywords internal
.codebook_prompt <- function(codebook) {
  lines <- vapply(seq_len(nrow(codebook)), function(i) {
    ex <- codebook$example[i]
    ex_txt <- if (is.na(ex) || !nzchar(ex)) "" else paste0(" Example: ", ex)
    paste0("- ", codebook$code[i], ": ", codebook$definition[i], ex_txt)
  }, character(1))
  paste(lines, collapse = "\n")
}

#' @title Apply a Codebook to Texts
#'
#' @description
#' Suggests codes for each unit of text from a supplied codebook using an AI
#' provider (OpenAI or Gemini). Documents are split into units (paragraphs by
#' default) before coding, and each unit receives zero or more codes. Output
#' is a set of suggestions for human confirmation, not final codes.
#'
#' @param texts Character vector of documents. Names become `doc_id`.
#' @param codebook Data frame with `code` and `definition` columns; optional
#'   `example` and `color`.
#' @param unit Unit of analysis: "paragraph" (default, split on blank lines),
#'   "sentence", or "document" (one unit per element of `texts`).
#' @param max_codes Maximum codes per unit (default 3). Units where no code
#'   fits return one row with `code = NA`.
#' @param provider AI provider: "auto" (default), "openai", or "gemini".
#' @param model Optional model id; provider default when NULL.
#' @param temperature Sampling temperature (default 0 for reproducibility).
#' @param api_key Optional API key; falls back to the provider env var.
#' @param delay Seconds to wait between provider calls (default 1).
#' @param verbose Logical; print per-unit progress (default TRUE).
#'
#' @return A tibble with one row per unit-code pair: `doc_id`, `unit_id`,
#'   `start`, `end` (character offsets of the unit within its document),
#'   `code`, `confidence`, `rationale`. Returns `invisible(NULL)` when no
#'   provider key is available.
#'
#' @seealso [code_retest()] for AI stability; [code_agreement()] for
#'   inter-coder reliability; [call_llm_api()] for the direct provider call.
#' @concept qualitative-coding
#' @export
apply_codes <- function(texts, codebook,
                        unit = c("paragraph", "sentence", "document"),
                        max_codes = 3,
                        provider = c("auto", "openai", "gemini"),
                        model = NULL, temperature = 0, api_key = NULL,
                        delay = 1, verbose = TRUE) {
  unit <- match.arg(unit)
  provider <- match.arg(provider)
  max_codes <- max(1L, as.integer(max_codes))
  if (!requireNamespace("httr", quietly = TRUE) ||
      !requireNamespace("jsonlite", quietly = TRUE)) {
    stop("The 'httr' and 'jsonlite' packages are required. ",
         "Install with install.packages(c('httr', 'jsonlite')).", call. = FALSE)
  }
  codebook <- .validate_codebook(codebook)
  doc_id <- if (!is.null(names(texts))) names(texts) else as.character(seq_along(texts))
  texts <- as.character(texts)

  units_tbl <- dplyr::bind_rows(lapply(seq_along(texts), function(i) {
    u <- .split_units(texts[i], unit)
    if (nrow(u) == 0) return(NULL)
    u$doc_id <- doc_id[i]
    u$unit_id <- if (unit == "document") doc_id[i] else paste0(doc_id[i], ".", seq_len(nrow(u)))
    u
  }))
  if (is.null(units_tbl) || nrow(units_tbl) == 0) {
    stop("texts contain no codable units.", call. = FALSE)
  }

  if (provider == "auto") {
    provider <- if (!is.null(api_key) && grepl("^sk-", api_key)) {
      "openai"
    } else if (!is.null(api_key) && grepl("^AIza", api_key)) {
      "gemini"
    } else if (nzchar(Sys.getenv("OPENAI_API_KEY"))) {
      "openai"
    } else if (nzchar(Sys.getenv("GEMINI_API_KEY"))) {
      "gemini"
    } else {
      message("No AI provider available. Set OPENAI_API_KEY or GEMINI_API_KEY.")
      return(invisible(NULL))
    }
  }
  if (is.null(api_key)) {
    api_key <- switch(provider,
      "openai" = Sys.getenv("OPENAI_API_KEY"),
      "gemini" = Sys.getenv("GEMINI_API_KEY"))
  }
  if (!nzchar(api_key)) return(.notify_missing_api_key(provider))

  system_prompt <- paste0(
    "Assign between zero and ", max_codes, " codes from the codebook to the text. ",
    "Codebook:\n", .codebook_prompt(codebook),
    "\n\nReturn ONLY a JSON object: {\"assignments\": [{\"code\": \"<code>\", ",
    "\"confidence\": <0-1>, \"rationale\": \"<short reason>\"}]}. ",
    "Use an empty array when no code fits.")

  valid_codes <- codebook$code
  n_units <- nrow(units_tbl)
  rows <- lapply(seq_len(n_units), function(i) {
    if (verbose) message("Coding unit ", i, " of ", n_units)
    parsed <- tryCatch({
      response <- call_llm_api(
        provider = provider, system_prompt = system_prompt,
        user_prompt = units_tbl$unit_text[i], model = model,
        temperature = temperature, max_tokens = 300, api_key = api_key)
      .parse_codes_response(response, valid_codes, max_codes)
    }, error = function(e) {
      tibble::tibble(code = NA_character_, confidence = NA_real_, rationale = NA_character_)
    })
    if (i < n_units) Sys.sleep(delay)
    tibble::tibble(
      doc_id = units_tbl$doc_id[i], unit_id = units_tbl$unit_id[i],
      start = units_tbl$start[i], end = units_tbl$end[i],
      code = parsed$code, confidence = parsed$confidence, rationale = parsed$rationale)
  })
  dplyr::bind_rows(rows)
}

#' @title AI Coding Retest Stability
#'
#' @description
#' Codes the same sample of texts more than once at identical settings and
#' reports how often the runs agree, next to a shuffled-label baseline. Low
#' temperature does not guarantee stable output, so retest agreement is
#' measured rather than assumed. The baseline shows the agreement expected if
#' codes were unrelated to the texts; retest agreement should sit well above
#' it.
#'
#' @param texts Character vector of documents. Names become `doc_id`.
#' @param codebook Data frame with `code` and `definition` columns.
#' @param n_runs Number of coding runs (default 2).
#' @param sample_n Maximum number of documents to sample (default 50).
#' @param seed Seed for document sampling and the shuffled baseline.
#' @param ... Passed to [apply_codes()] (`unit`, `max_codes`, `provider`,
#'   `api_key`, `delay`, ...).
#'
#' @return A list with `summary` (tibble of metric, estimate, n_units,
#'   n_runs) and `runs` (all assignments, with a `run` column), or
#'   `invisible(NULL)` when fewer than two runs complete.
#'
#' @seealso [apply_codes()]; [code_agreement()] for human inter-coder
#'   reliability, which this does not replace.
#' @concept qualitative-coding
#' @export
code_retest <- function(texts, codebook, n_runs = 2, sample_n = 50, seed = 123, ...) {
  doc_id <- if (!is.null(names(texts))) names(texts) else as.character(seq_along(texts))
  idx <- withr::with_seed(seed, sample(seq_along(texts), min(sample_n, length(texts))))
  sampled <- stats::setNames(as.character(texts)[idx], doc_id[idx])

  runs <- lapply(seq_len(n_runs), function(r) {
    out <- apply_codes(sampled, codebook, ...)
    if (is.null(out) || nrow(out) == 0) NULL else dplyr::mutate(out, run = r)
  })
  runs <- Filter(Negate(is.null), runs)
  if (length(runs) < 2) {
    message("code_retest needs at least two completed runs.")
    return(invisible(NULL))
  }

  sigs <- lapply(runs, function(d) {
    d %>%
      dplyr::mutate(code = dplyr::coalesce(.data$code, "")) %>%
      dplyr::group_by(.data$unit_id) %>%
      dplyr::summarise(
        sig = paste(sort(unique(.data$code[nzchar(.data$code)])), collapse = "+"),
        .groups = "drop")
  })

  pairs <- utils::combn(length(sigs), 2, simplify = FALSE)
  pair_stats <- vapply(pairs, function(p) {
    m <- dplyr::inner_join(sigs[[p[1]]], sigs[[p[2]]], by = "unit_id", suffix = c("_1", "_2"))
    shuffled <- withr::with_seed(seed + p[2], sample(m$sig_2))
    c(retest = mean(m$sig_1 == m$sig_2),
      baseline = mean(m$sig_1 == shuffled),
      n = nrow(m))
  }, numeric(3))

  list(
    summary = tibble::tibble(
      metric = c("retest_agreement", "shuffled_baseline"),
      estimate = c(mean(pair_stats["retest", ]), mean(pair_stats["baseline", ])),
      n_units = as.integer(round(mean(pair_stats["n", ]))),
      n_runs = length(runs)),
    runs = dplyr::bind_rows(runs)
  )
}

#' @keywords internal
.qc_ratings <- function(assignments, units = c("intersection", "union")) {
  units <- match.arg(units)
  miss <- setdiff(c("doc_id", "code", "coder"), names(assignments))
  if (length(miss) > 0) {
    stop("assignments is missing column(s): ", paste(miss, collapse = ", "), call. = FALSE)
  }
  a <- assignments[!duplicated(assignments[c("doc_id", "coder")]), c("doc_id", "code", "coder")]
  wide <- tidyr::pivot_wider(a, id_cols = "doc_id", names_from = "coder", values_from = "code")
  m <- as.matrix(wide[, -1, drop = FALSE])
  rownames(m) <- wide$doc_id
  storage.mode(m) <- "character"
  if (units == "intersection") m <- m[stats::complete.cases(m), , drop = FALSE]
  m
}

#' @keywords internal
.percent_agreement <- function(ratings) {
  cc <- ratings[stats::complete.cases(ratings), , drop = FALSE]
  if (nrow(cc) == 0) return(NA_real_)
  mean(apply(cc, 1, function(r) length(unique(r)) == 1))
}

#' @keywords internal
.pabak <- function(ratings) {
  if (ncol(ratings) != 2) return(NA_real_)
  po <- .percent_agreement(ratings)
  if (is.na(po)) return(NA_real_)
  2 * po - 1
}

#' @keywords internal
.gwet_ac1 <- function(ratings) {
  cats <- sort(unique(as.vector(ratings[!is.na(ratings)])))
  q <- length(cats)
  if (q < 2) return(if (q == 1) 1 else NA_real_)
  ri <- rowSums(!is.na(ratings))
  use <- ri >= 2
  if (!any(use)) return(NA_real_)
  R <- ratings[use, , drop = FALSE]
  ri <- ri[use]
  pa_i <- vapply(seq_len(nrow(R)), function(i) {
    counts <- table(factor(R[i, !is.na(R[i, ])], levels = cats))
    sum(counts * (counts - 1)) / (ri[i] * (ri[i] - 1))
  }, numeric(1))
  pa <- mean(pa_i)
  pik <- vapply(cats, function(k) {
    mean(vapply(seq_len(nrow(R)), function(i) sum(R[i, ] == k, na.rm = TRUE) / ri[i], numeric(1)))
  }, numeric(1))
  pe <- sum(pik * (1 - pik)) / (q - 1)
  (pa - pe) / (1 - pe)
}

#' @keywords internal
.kappa_estimate <- function(ratings) {
  cc <- ratings[stats::complete.cases(ratings), , drop = FALSE]
  if (nrow(cc) < 1) return(NA_real_)
  res <- if (ncol(cc) == 2) {
    tryCatch(irr::kappa2(cc), error = function(e) NULL)
  } else {
    tryCatch(irr::kappam.fleiss(cc), error = function(e) NULL)
  }
  if (is.null(res) || !is.finite(res$value)) NA_real_ else res$value
}

#' @keywords internal
.alpha_estimate <- function(ratings) {
  coded <- matrix(as.integer(factor(ratings)), nrow = nrow(ratings))
  res <- tryCatch(irr::kripp.alpha(t(coded), method = "nominal"), error = function(e) NULL)
  if (is.null(res) || !is.finite(res$value)) NA_real_ else res$value
}

#' @keywords internal
.agreement_metrics <- function(ratings, metrics) {
  vals <- list()
  if ("percent" %in% metrics) vals$percent <- .percent_agreement(ratings)
  if ("pabak" %in% metrics)   vals$pabak   <- .pabak(ratings)
  if ("ac1" %in% metrics)     vals$ac1     <- .gwet_ac1(ratings)
  if ("kappa" %in% metrics)   vals$kappa   <- .kappa_estimate(ratings)
  if ("alpha" %in% metrics)   vals$alpha   <- .alpha_estimate(ratings)
  tibble::tibble(metric = names(vals),
                 estimate = unlist(vals, use.names = FALSE),
                 n = nrow(ratings))
}

#' @keywords internal
.agreement_by_code <- function(ratings, metrics) {
  codes <- sort(unique(as.vector(ratings[!is.na(ratings)])))
  rows <- lapply(codes, function(k) {
    bin <- ifelse(is.na(ratings), NA, ifelse(ratings == k, "yes", "no"))
    m <- .agreement_metrics(bin, metrics)
    m$code <- k
    m[, c("code", "metric", "estimate", "n")]
  })
  dplyr::bind_rows(rows)
}

#' @keywords internal
.qc_disagreements <- function(ratings) {
  differ <- apply(ratings, 1, function(r) {
    r <- r[!is.na(r)]
    length(unique(r)) > 1
  })
  d <- ratings[differ, , drop = FALSE]
  ids <- rownames(d)
  tibble::as_tibble(cbind(
    data.frame(doc_id = if (is.null(ids)) character(0) else ids, stringsAsFactors = FALSE),
    as.data.frame(d, stringsAsFactors = FALSE)
  ))
}

#' @keywords internal
.span_coverage <- function(assignments, by_code = TRUE) {
  need <- c("doc_id", "code", "coder", "start", "end")
  miss <- setdiff(need, names(assignments))
  if (length(miss) > 0) {
    stop("align = 'coverage' requires column(s): ", paste(miss, collapse = ", "), call. = FALSE)
  }
  a <- assignments[!is.na(assignments$code), need]
  coders <- unique(a$coder)
  if (length(coders) < 2) {
    stop("At least two coders are required.", call. = FALSE)
  }
  pairs <- expand.grid(coder = coders, other = coders, stringsAsFactors = FALSE)
  pairs <- pairs[pairs$coder != pairs$other, ]
  cov_df <- dplyr::bind_rows(lapply(seq_len(nrow(pairs)), function(i) {
    A <- a[a$coder == pairs$coder[i], , drop = FALSE]
    B <- a[a$coder == pairs$other[i], , drop = FALSE]
    hit <- vapply(seq_len(nrow(A)), function(j) {
      b <- B[B$doc_id == A$doc_id[j] & B$code == A$code[j], , drop = FALSE]
      any(b$start <= A$end[j] & A$start[j] <= b$end)
    }, logical(1))
    A$other <- pairs$other[i]
    A$covered <- hit
    A
  }))
  overall <- cov_df %>%
    dplyr::group_by(.data$coder, .data$other) %>%
    dplyr::summarise(metric = "coverage", estimate = mean(.data$covered),
                     n = dplyr::n(), .groups = "drop")
  by_code_tbl <- if (by_code) {
    cov_df %>%
      dplyr::group_by(.data$code, .data$coder, .data$other) %>%
      dplyr::summarise(metric = "coverage", estimate = mean(.data$covered),
                       n = dplyr::n(), .groups = "drop")
  } else {
    NULL
  }
  disagree <- cov_df %>%
    dplyr::filter(!.data$covered) %>%
    dplyr::select("doc_id", "coder", "code", "start", "end", "other")
  list(overall = overall, by_code = by_code_tbl, disagree = tibble::as_tibble(disagree))
}

#' @title Inter-Coder Agreement From Code Assignments
#'
#' @description
#' Chance-corrected agreement between two or more coders' code assignments.
#' Reports Krippendorff's alpha and Cohen's/Fleiss' kappa (via irr) plus percent
#' agreement, Gwet's AC1, and PABAK computed inline. AC1 and PABAK stay stable
#' under skewed code prevalence, where kappa collapses (the kappa paradox).
#'
#' With `align = "grid"` (default), coders are assumed to share units: rows
#' pivot on `unit_id` when present, else `doc_id`. When a coder assigned more
#' than one code to a unit, the highest-confidence code is kept (first row
#' when no `confidence` column exists); per-code multi-label agreement is in
#' `by_code`. With `align = "coverage"`, coders may have different span
#' boundaries: for each ordered coder pair, the proportion of one coder's
#' spans that overlap a same-code span from the other is reported instead
#' (chance-corrected metrics do not apply across unaligned units, so
#' `metrics` and `units` are ignored).
#'
#' @param assignments Data frame with `doc_id`, `code`, and `coder` columns.
#'   `unit_id` and `confidence` are used when present. Coverage additionally
#'   requires `start` and `end`.
#' @param metrics Metrics to report: any of "alpha", "kappa", "ac1", "pabak",
#'   "percent" (all by default). Grid alignment only.
#' @param units "intersection" (default, units coded by every coder) or
#'   "union" (uncoded units count as missing). Grid alignment only.
#' @param by_code Logical; also report per-code agreement.
#' @param align "grid" (default) or "coverage".
#'
#' @return A list with `overall`, `by_code` (NULL when `by_code` is FALSE),
#'   and `disagree`. For grid alignment these are the metric, per-code, and
#'   disagreement tables; for coverage they hold per-coder-pair span coverage
#'   and the uncovered spans.
#'
#' @seealso [apply_codes()] to produce assignments; [merge_codes()] to combine
#'   coders; [code_retest()] for AI stability.
#' @concept qualitative-coding
#' @export
code_agreement <- function(assignments,
                           metrics = c("alpha", "kappa", "ac1", "pabak", "percent"),
                           units = c("intersection", "union"),
                           by_code = TRUE,
                           align = c("grid", "coverage")) {
  metrics <- match.arg(metrics, several.ok = TRUE)
  units <- match.arg(units)
  align <- match.arg(align)
  if (align == "coverage") return(.span_coverage(assignments, by_code))
  if (!requireNamespace("irr", quietly = TRUE)) {
    stop("Package 'irr' is required. Install with install.packages('irr').", call. = FALSE)
  }
  a <- assignments
  if ("unit_id" %in% names(a)) a$doc_id <- a$unit_id
  if ("confidence" %in% names(a)) {
    a <- a[order(a$confidence, decreasing = TRUE, na.last = TRUE), ]
  }
  ratings <- .qc_ratings(a, units)
  if (nrow(ratings) == 0 || ncol(ratings) < 2) {
    stop("At least two coders with shared units are required.", call. = FALSE)
  }
  list(
    overall = .agreement_metrics(ratings, metrics),
    by_code = if (by_code) .agreement_by_code(ratings, metrics) else NULL,
    disagree = .qc_disagreements(ratings)
  )
}

#' @title Combine Coder Assignment Files
#'
#' @description
#' Reads per-coder assignment files (each a saved assignments tibble) and binds
#' them into one long assignments table for [code_agreement()]. Supports the
#' async workflow where each coder codes a copy and exports it.
#'
#' @param files Character vector of `.rds` paths, or a list of data frames.
#'
#' @return A tibble of combined, de-duplicated assignments.
#'
#' @seealso [code_agreement()] for the agreement summary.
#' @concept qualitative-coding
#' @export
merge_codes <- function(files) {
  if (is.data.frame(files)) files <- list(files)
  parts <- lapply(files, function(f) if (is.data.frame(f)) f else readRDS(f))
  dplyr::distinct(dplyr::bind_rows(parts))
}
