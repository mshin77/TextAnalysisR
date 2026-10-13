fake_model <- list(transform = function(texts, embeddings = NULL) list(c(-1L, 0L, 2L)[seq_along(texts)]))

test_that("transform_topic_model shifts topic IDs to the fit convention", {
  out <- transform_topic_model(list(bertopic_model = fake_model), c("a", "b", "c"))
  expect_equal(out$topic, c(0L, 1L, 3L))
})

test_that("transform_topic_model returns NA for blank and missing texts", {
  out <- transform_topic_model(list(bertopic_model = fake_model), c("a", NA, " ", "b"))
  expect_equal(out$topic, c(0L, NA, NA, 1L))
})

test_that("transform_topic_model rejects bad input", {
  fit <- list(bertopic_model = fake_model)
  expect_error(transform_topic_model(list(), "a"), "no BERTopic model")
  expect_error(transform_topic_model(fit, 1:2), "character")
  expect_error(transform_topic_model(fit, c("a", "b"), embeddings = matrix(0, 1, 2)), "one row")
})

test_that("save_topic_model refuses to overwrite without overwrite = TRUE", {
  path <- withr::local_tempdir()
  saveRDS(list(), file.path(path, "fit.rds"))
  expect_error(save_topic_model(list(bertopic_model = fake_model, embedding_model = "m"), path), "overwrite")
})

test_that("save_topic_model requires an embedding model name", {
  expect_error(save_topic_model(list(bertopic_model = fake_model), withr::local_tempdir()), "embedding model")
})

test_that("load_topic_model rejects a folder without fit.rds", {
  expect_error(load_topic_model(withr::local_tempdir()), "not a saved topic model")
})

test_that("code_provenance returns recorded settings and rejects plain input", {
  x <- tibble::tibble(a = 1)
  attr(x, "llm") <- list(provider = "openai", model = NULL, temperature = 0, prompt_hash = "h")
  out <- code_provenance(x)
  expect_equal(nrow(out), 1L)
  expect_equal(out$prompt_hash, "h")
  expect_true(is.na(out$model))
  expect_error(code_provenance(tibble::tibble(a = 1)), "no recorded settings")
})

test_that("apply_codes records seed and served model from the provider", {
  fake <- function(...) {
    out <- '{"assignments": [{"code": "a", "confidence": 0.9, "rationale": "r"}]}'
    attr(out, "served_model") <- "gpt-4.1-mini-2025-04-14"
    attr(out, "system_fingerprint") <- "fp_1"
    out
  }
  testthat::local_mocked_bindings(call_llm_api = fake, .resolve_provider = function(...) list(provider = "openai", api_key = "k"))
  book <- data.frame(code = "a", definition = "d")
  res <- apply_codes(c(d1 = "text one", d2 = "text two"), book, unit = "document", seed = 7, delay = 0, verbose = FALSE)
  prov <- code_provenance(res)
  expect_equal(prov$seed, 7)
  expect_equal(prov$served_model, "gpt-4.1-mini-2025-04-14")
  expect_equal(prov$system_fingerprint, "fp_1")
})
