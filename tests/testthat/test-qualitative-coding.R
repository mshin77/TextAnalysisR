test_that(".validate_codebook normalizes a minimal codebook", {
  cb <- tibble::tibble(code = c("a", "b"), definition = c("def a", "def b"))
  out <- TextAnalysisR:::.validate_codebook(cb)
  expect_named(out, c("code", "definition", "example", "color"))
  expect_true(all(is.na(out$example)))
  expect_equal(out$code, c("a", "b"))
})

test_that(".validate_codebook errors on missing code column", {
  expect_error(
    TextAnalysisR:::.validate_codebook(tibble::tibble(definition = "x")),
    "code"
  )
})

test_that(".validate_codebook errors on duplicate codes", {
  cb <- tibble::tibble(code = c("a", "a"), definition = c("d1", "d2"))
  expect_error(TextAnalysisR:::.validate_codebook(cb), "duplicate")
})

test_that(".split_units splits paragraphs on blank lines with verbatim offsets", {
  text <- "First paragraph here.\n\nSecond one.\n \nThird."
  u <- TextAnalysisR:::.split_units(text, "paragraph")
  expect_equal(nrow(u), 3)
  expect_equal(u$unit_text, c("First paragraph here.", "Second one.", "Third."))
  for (i in seq_len(nrow(u))) {
    expect_equal(substr(text, u$start[i], u$end[i]), u$unit_text[i])
  }
})

test_that(".split_units splits sentences and keeps offsets verbatim", {
  text <- "One sentence. Another one! A third?"
  u <- TextAnalysisR:::.split_units(text, "sentence")
  expect_equal(u$unit_text, c("One sentence.", "Another one!", "A third?"))
  for (i in seq_len(nrow(u))) {
    expect_equal(substr(text, u$start[i], u$end[i]), u$unit_text[i])
  }
})

test_that(".split_units returns one full-span unit for document mode", {
  u <- TextAnalysisR:::.split_units("Whole text.", "document")
  expect_equal(nrow(u), 1)
  expect_equal(u$start, 1L)
  expect_equal(u$end, nchar("Whole text."))
})

test_that(".split_units returns zero rows for empty or NA text", {
  expect_equal(nrow(TextAnalysisR:::.split_units("", "paragraph")), 0)
  expect_equal(nrow(TextAnalysisR:::.split_units("  \n ", "sentence")), 0)
  expect_equal(nrow(TextAnalysisR:::.split_units(NA_character_, "paragraph")), 0)
})

test_that(".parse_codes_response parses an assignments array", {
  skip_if_not_installed("jsonlite")
  r <- '{"assignments": [{"code": "a", "confidence": 0.8, "rationale": "mentions a"},
                         {"code": "b", "confidence": 0.5, "rationale": "hints b"}]}'
  out <- TextAnalysisR:::.parse_codes_response(r, valid_codes = c("a", "b"), max_codes = 3)
  expect_equal(out$code, c("a", "b"))
  expect_equal(out$confidence, c(0.8, 0.5))
  expect_equal(out$rationale, c("mentions a", "hints b"))
})

test_that(".parse_codes_response enforces max_codes and drops invalid codes", {
  skip_if_not_installed("jsonlite")
  r <- '{"assignments": [{"code": "z", "confidence": 0.9},
                         {"code": "a", "confidence": 0.8},
                         {"code": "b", "confidence": 0.7},
                         {"code": "c", "confidence": 0.6}]}'
  out <- TextAnalysisR:::.parse_codes_response(r, valid_codes = c("a", "b", "c"), max_codes = 2)
  expect_equal(out$code, c("a", "b"))
})

test_that(".parse_codes_response clamps confidence and dedupes codes", {
  skip_if_not_installed("jsonlite")
  r <- '{"assignments": [{"code": "a", "confidence": 1.7}, {"code": "a", "confidence": 0.2}]}'
  out <- TextAnalysisR:::.parse_codes_response(r, valid_codes = "a", max_codes = 3)
  expect_equal(nrow(out), 1)
  expect_equal(out$confidence, 1)
})

test_that(".parse_codes_response returns one NA row for empty array or bad input", {
  skip_if_not_installed("jsonlite")
  empty <- TextAnalysisR:::.parse_codes_response('{"assignments": []}', "a", 3)
  expect_equal(nrow(empty), 1)
  expect_true(is.na(empty$code))
  bad <- TextAnalysisR:::.parse_codes_response("not json", "a", 3)
  expect_true(is.na(bad$code))
  url <- TextAnalysisR:::.parse_codes_response("https://example.org/x.json", "a", 3)
  expect_true(is.na(url$code))
})

test_that(".parse_codes_response strips markdown code fences", {
  skip_if_not_installed("jsonlite")
  r <- "```json\n{\"assignments\": [{\"code\": \"a\", \"confidence\": 0.6}]}\n```"
  out <- TextAnalysisR:::.parse_codes_response(r, valid_codes = "a", max_codes = 3)
  expect_equal(out$code, "a")
})

test_that(".codebook_prompt lists every code and definition", {
  cb <- tibble::tibble(code = c("a", "b"), definition = c("def a", "def b"),
                       example = c("ex a", NA), color = NA_character_)
  txt <- TextAnalysisR:::.codebook_prompt(cb)
  expect_true(grepl("a", txt) && grepl("def a", txt))
  expect_true(grepl("b", txt) && grepl("def b", txt))
})

.qc_mock_response <- '{"assignments": [{"code": "a", "confidence": 0.7, "rationale": "ok"}]}'

test_that("apply_codes returns one row per unit-code pair with a mocked provider", {
  skip_if_not_installed("httr")
  skip_if_not_installed("jsonlite")
  cb <- tibble::tibble(code = c("a", "b"), definition = c("def a", "def b"))
  fake <- function(...) .qc_mock_response
  testthat::local_mocked_bindings(call_llm_api = fake, .package = "TextAnalysisR")
  out <- apply_codes(
    texts = c(d1 = "Para one.\n\nPara two.", d2 = "Only one."),
    codebook = cb, provider = "openai", api_key = "sk-test",
    delay = 0, verbose = FALSE
  )
  expect_named(out, c("doc_id", "unit_id", "start", "end", "code", "confidence", "rationale"))
  expect_equal(nrow(out), 3)
  expect_equal(out$unit_id, c("d1.1", "d1.2", "d2.1"))
  expect_equal(out$code, rep("a", 3))
  expect_equal(substr("Para one.\n\nPara two.", out$start[2], out$end[2]), "Para two.")
})

test_that("apply_codes with unit = 'document' keys units by doc_id", {
  skip_if_not_installed("httr")
  skip_if_not_installed("jsonlite")
  cb <- tibble::tibble(code = "a", definition = "def a")
  fake <- function(...) .qc_mock_response
  testthat::local_mocked_bindings(call_llm_api = fake, .package = "TextAnalysisR")
  out <- apply_codes(texts = c(d1 = "text one", d2 = "text two"), codebook = cb,
                     unit = "document", provider = "openai", api_key = "sk-test",
                     delay = 0, verbose = FALSE)
  expect_equal(out$unit_id, c("d1", "d2"))
})

test_that("apply_codes emits multiple rows per unit when the provider returns several codes", {
  skip_if_not_installed("httr")
  skip_if_not_installed("jsonlite")
  cb <- tibble::tibble(code = c("a", "b"), definition = c("def a", "def b"))
  fake <- function(...) {
    '{"assignments": [{"code": "a", "confidence": 0.8}, {"code": "b", "confidence": 0.4}]}'
  }
  testthat::local_mocked_bindings(call_llm_api = fake, .package = "TextAnalysisR")
  out <- apply_codes(texts = c(d1 = "text"), codebook = cb, unit = "document",
                     provider = "openai", api_key = "sk-test", delay = 0, verbose = FALSE)
  expect_equal(nrow(out), 2)
  expect_equal(out$unit_id, c("d1", "d1"))
  expect_equal(out$code, c("a", "b"))
})

test_that("apply_codes resolves provider from the explicit api_key before env vars", {
  skip_if_not_installed("httr")
  skip_if_not_installed("jsonlite")
  cb <- tibble::tibble(code = "a", definition = "def a")
  seen <- NULL
  fake <- function(provider, ...) {
    seen <<- provider
    .qc_mock_response
  }
  testthat::local_mocked_bindings(call_llm_api = fake, .package = "TextAnalysisR")
  withr::with_envvar(c(OPENAI_API_KEY = "sk-env"), {
    apply_codes(texts = "t", codebook = cb, api_key = "AIzaTest",
                delay = 0, verbose = FALSE)
  })
  expect_equal(seen, "gemini")
})

test_that("apply_codes returns NULL when no provider key is available", {
  skip_if_not_installed("httr")
  skip_if_not_installed("jsonlite")
  cb <- tibble::tibble(code = "a", definition = "def a")
  withr::with_envvar(c(OPENAI_API_KEY = "", GEMINI_API_KEY = ""), {
    expect_null(suppressMessages(
      apply_codes(texts = "t", codebook = cb, provider = "auto", verbose = FALSE)
    ))
  })
})

test_that("apply_codes rejects an unknown provider or unit", {
  skip_if_not_installed("httr")
  skip_if_not_installed("jsonlite")
  cb <- tibble::tibble(code = "a", definition = "def a")
  expect_error(apply_codes(texts = "t", codebook = cb, provider = "cohere"))
  expect_error(apply_codes(texts = "t", codebook = cb, unit = "line"))
})

test_that("apply_codes prints per-unit progress when verbose", {
  skip_if_not_installed("httr")
  skip_if_not_installed("jsonlite")
  cb <- tibble::tibble(code = "a", definition = "def a")
  fake <- function(...) .qc_mock_response
  testthat::local_mocked_bindings(call_llm_api = fake, .package = "TextAnalysisR")
  expect_message(
    apply_codes(texts = "t", codebook = cb, provider = "openai",
                api_key = "sk-test", delay = 0, verbose = TRUE),
    "Coding unit 1 of 1"
  )
})

test_that("code_retest reports perfect agreement for a deterministic mock", {
  skip_if_not_installed("httr")
  skip_if_not_installed("jsonlite")
  cb <- tibble::tibble(code = c("a", "b"), definition = c("def a", "def b"))
  fake <- function(provider, system_prompt, user_prompt, ...) {
    if (grepl("math", user_prompt)) {
      '{"assignments": [{"code": "a", "confidence": 0.9}]}'
    } else {
      '{"assignments": [{"code": "b", "confidence": 0.9}]}'
    }
  }
  testthat::local_mocked_bindings(call_llm_api = fake, .package = "TextAnalysisR")
  texts <- c(d1 = "math anxiety text", d2 = "reading text", d3 = "more math here",
             d4 = "reading again")
  res <- code_retest(texts, cb, n_runs = 2, sample_n = 4, seed = 1,
                     unit = "document", provider = "openai", api_key = "sk-test",
                     delay = 0, verbose = FALSE)
  est <- function(m) res$summary$estimate[res$summary$metric == m]
  expect_equal(est("retest_agreement"), 1)
  expect_true(est("shuffled_baseline") >= 0 && est("shuffled_baseline") <= 1)
  expect_equal(res$summary$n_units[1], 4L)
  expect_setequal(unique(res$runs$run), c(1, 2))
})

test_that("code_retest samples at most sample_n documents", {
  skip_if_not_installed("httr")
  skip_if_not_installed("jsonlite")
  cb <- tibble::tibble(code = "a", definition = "def a")
  fake <- function(...) .qc_mock_response
  testthat::local_mocked_bindings(call_llm_api = fake, .package = "TextAnalysisR")
  texts <- stats::setNames(paste("text", 1:10), paste0("d", 1:10))
  res <- code_retest(texts, cb, n_runs = 2, sample_n = 3, seed = 1,
                     unit = "document", provider = "openai", api_key = "sk-test",
                     delay = 0, verbose = FALSE)
  expect_equal(res$summary$n_units[1], 3L)
})

.qc_assignments <- function() {
  tibble::tibble(
    doc_id = rep(c("d1", "d2", "d3", "d4"), each = 2),
    code   = c("a", "a", "a", "a", "a", "a", "a", "b"),
    coder  = rep(c("c1", "c2"), times = 4)
  )
}

test_that("code_agreement reports percent, PABAK and AC1 by hand values", {
  skip_if_not_installed("irr")
  res <- code_agreement(.qc_assignments())
  est <- function(m) res$overall$estimate[res$overall$metric == m]
  expect_equal(est("percent"), 0.75)
  expect_equal(est("pabak"), 0.5)
  expect_equal(est("ac1"), 0.68, tolerance = 0.01)
})

test_that("code_agreement exposes the kappa paradox (kappa 0 while agreement high)", {
  skip_if_not_installed("irr")
  res <- code_agreement(.qc_assignments())
  kappa <- res$overall$estimate[res$overall$metric == "kappa"]
  expect_equal(kappa, 0, tolerance = 1e-8)
})

test_that("code_agreement lists only disagreeing units", {
  skip_if_not_installed("irr")
  res <- code_agreement(.qc_assignments())
  expect_equal(res$disagree$doc_id, "d4")
})

test_that("code_agreement returns per-code rows", {
  skip_if_not_installed("irr")
  res <- code_agreement(.qc_assignments())
  expect_setequal(unique(res$by_code$code), c("a", "b"))
})

test_that("code_agreement is 1 on perfect agreement", {
  skip_if_not_installed("irr")
  perfect <- tibble::tibble(
    doc_id = rep(c("d1", "d2"), each = 2),
    code   = "a",
    coder  = rep(c("c1", "c2"), times = 2)
  )
  res <- code_agreement(perfect)
  est <- function(m) res$overall$estimate[res$overall$metric == m]
  expect_equal(est("percent"), 1)
  expect_equal(est("pabak"), 1)
  expect_equal(est("ac1"), 1)
  expect_named(res$disagree, c("doc_id", "c1", "c2"))
  expect_equal(nrow(res$disagree), 0)
  expect_true(is.na(res$overall$estimate[res$overall$metric == "kappa"]))
})

test_that("code_agreement pivots on unit_id when present", {
  skip_if_not_installed("irr")
  a <- tibble::tibble(
    doc_id  = "d1",
    unit_id = rep(c("d1.1", "d1.2"), each = 2),
    code    = c("a", "a", "a", "b"),
    coder   = rep(c("c1", "c2"), times = 2)
  )
  res <- code_agreement(a)
  expect_equal(res$overall$n[1], 2)
  expect_equal(res$disagree$doc_id, "d1.2")
})

test_that("code_agreement keeps the highest-confidence code per unit and coder", {
  skip_if_not_installed("irr")
  a <- tibble::tibble(
    doc_id     = rep("d1", 3),
    code       = c("b", "a", "a"),
    coder      = c("c1", "c1", "c2"),
    confidence = c(0.4, 0.9, 0.8)
  )
  res <- code_agreement(a)
  expect_equal(res$overall$estimate[res$overall$metric == "percent"], 1)
})

test_that("code_agreement coverage reports span overlap per coder pair", {
  a <- tibble::tibble(
    doc_id = c("d1", "d1", "d1"),
    coder  = c("c1", "c1", "c2"),
    code   = c("a", "b", "a"),
    start  = c(1, 200, 50),
    end    = c(100, 300, 80)
  )
  res <- code_agreement(a, align = "coverage")
  c1_by_c2 <- res$overall$estimate[res$overall$coder == "c1" & res$overall$other == "c2"]
  c2_by_c1 <- res$overall$estimate[res$overall$coder == "c2" & res$overall$other == "c1"]
  expect_equal(c1_by_c2, 0.5)
  expect_equal(c2_by_c1, 1)
  expect_equal(res$disagree$code, "b")
})

test_that("code_agreement coverage errors without span columns", {
  expect_error(
    code_agreement(.qc_assignments(), align = "coverage"),
    "start"
  )
})

test_that("code_agreement errors without the required columns", {
  skip_if_not_installed("irr")
  expect_error(code_agreement(tibble::tibble(doc_id = "d1", code = "a")), "coder")
})

test_that("merge_codes binds per-coder assignment files", {
  dir <- withr::local_tempdir()
  a1 <- tibble::tibble(doc_id = "d1", code = "a", coder = "c1")
  a2 <- tibble::tibble(doc_id = "d1", code = "b", coder = "c2")
  f1 <- file.path(dir, "c1.rds"); saveRDS(a1, f1)
  f2 <- file.path(dir, "c2.rds"); saveRDS(a2, f2)
  out <- merge_codes(c(f1, f2))
  expect_equal(nrow(out), 2)
  expect_setequal(out$coder, c("c1", "c2"))
})

test_that("merge_codes accepts a single bare data frame", {
  a1 <- tibble::tibble(doc_id = "d1", code = "a", coder = "c1")
  out <- merge_codes(a1)
  expect_equal(nrow(out), 1)
})

.qc_authored <- function() {
  tibble::tibble(
    doc_id = rep(c("d1", "d2", "d3", "d4"), each = 3),
    code   = c("a", "a", "a",
               "a", "a", "b",
               "a", "b", "a",
               "b", "b", "b"),
    coder  = rep(c("author", "c2", "c3"), times = 4)
  )
}

test_that("code_agreement returns NULL independent without codebook_authors", {
  skip_if_not_installed("irr")
  res <- code_agreement(.qc_assignments())
  expect_null(res$independent)
})

test_that("code_agreement scores the non-author coders separately", {
  skip_if_not_installed("irr")
  res <- code_agreement(.qc_authored(), codebook_authors = "author")
  expect_s3_class(res$independent, "tbl_df")
  expect_setequal(res$independent$metric, res$overall$metric)
  expect_true(all(res$independent$n == 4))
})

test_that("code_agreement drops independent when fewer than two others remain", {
  skip_if_not_installed("irr")
  res <- code_agreement(.qc_assignments(), codebook_authors = "c1")
  expect_null(res$independent)
})

test_that("code_agreement ignores codebook_authors under coverage alignment", {
  a <- tibble::tibble(
    doc_id = c("d1", "d1"),
    code   = c("a", "a"),
    coder  = c("c1", "c2"),
    start  = c(1L, 3L),
    end    = c(10L, 12L)
  )
  res <- code_agreement(a, align = "coverage", codebook_authors = "c1")
  expect_named(res, c("overall", "by_code", "disagree"))
})

.qc_coded <- function() {
  tibble::tibble(
    doc_id  = c("d1", "d1", "d2"),
    unit_id = c("d1.1", "d1.2", "d2.1"),
    start   = c(1L, 13L, 1L),
    end     = c(11L, 22L, 6L),
    code    = c("a", NA_character_, NA_character_)
  )
}

test_that("uncoded_units returns the uncoded units with their text", {
  texts <- c(d1 = "first unit. second one", d2 = "lonely")
  out <- uncoded_units(.qc_coded(), texts)
  expect_equal(out$unit_id, c("d1.2", "d2.1"))
  expect_equal(out$unit_text, c("second one", "lonely"))
  expect_type(out$start, "integer")
})

test_that("uncoded_units keeps a unit only when every code is NA", {
  a <- tibble::tibble(
    doc_id  = c("d1", "d1"),
    unit_id = c("d1.1", "d1.1"),
    start   = c(1L, 1L),
    end     = c(5L, 5L),
    code    = c("a", NA_character_)
  )
  expect_equal(nrow(uncoded_units(a, c(d1 = "hello"))), 0L)
})

test_that("uncoded_units returns zero rows when every unit is coded", {
  a <- .qc_coded()
  a$code <- c("a", "b", "c")
  out <- uncoded_units(a, c(d1 = "first unit. second one", d2 = "lonely"))
  expect_equal(nrow(out), 0L)
  expect_named(out, c("doc_id", "unit_id", "start", "end", "unit_text"))
})

test_that("uncoded_units keys unnamed texts by position", {
  a <- tibble::tibble(doc_id = "1", unit_id = "1.1", start = 1L, end = 5L,
                      code = NA_character_)
  expect_equal(uncoded_units(a, "hello there")$unit_text, "hello")
})

test_that("uncoded_units errors on missing columns", {
  expect_error(uncoded_units(tibble::tibble(doc_id = "d1"), c(d1 = "x")),
               "missing column")
})

test_that("merge_codes reads csv and rds paths by extension", {
  a <- tibble::tibble(doc_id = "d1", code = "a", coder = "c1")
  b <- tibble::tibble(doc_id = "d1", code = "a", coder = "c2")
  p_rds <- withr::local_tempfile(fileext = ".rds")
  p_csv <- withr::local_tempfile(fileext = ".csv")
  saveRDS(a, p_rds)
  utils::write.csv(b, p_csv, row.names = FALSE)
  out <- merge_codes(c(p_rds, p_csv))
  expect_equal(nrow(out), 2L)
  expect_setequal(out$coder, c("c1", "c2"))
})

test_that("merge_codes rejects an unsupported extension", {
  p <- withr::local_tempfile(fileext = ".docx")
  file.create(p)
  expect_error(merge_codes(p), "Unsupported coder file type")
})
