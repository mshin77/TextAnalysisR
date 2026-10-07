keyness_fixture <- function() {
  texts <- c(
    rep("flipped classroom dissertation study of student anxiety and self efficacy", 6),
    rep("journal article reports software tools for reading instruction in schools", 6)
  )
  dfm <- quanteda::dfm(quanteda::tokens(texts))
  list(dfm = dfm, target = rep(c(TRUE, FALSE), each = 6))
}

test_that("extract_keywords_keyness reports counts, rates, effect size and adjusted p", {
  skip_if_not_installed("quanteda.textstats")
  fx <- keyness_fixture()
  k <- extract_keywords_keyness(fx$dfm, target = fx$target, top_n = 50)

  expect_named(k, c("Keyword", "Keyness_Score", "Target_Count", "Reference_Count",
                    "Target_per_10k", "Reference_per_10k", "Log_Ratio", "P_Adjusted"))
  row <- k[k$Keyword == "dissertation", ]
  expect_equal(row$Target_Count, 6)
  expect_equal(row$Reference_Count, 0)
  expect_gt(row$Log_Ratio, 0)
  expect_true(all(k$P_Adjusted >= 0 & k$P_Adjusted <= 1))
  expect_lt(k$Log_Ratio[k$Keyword == "software"], 0)
})

test_that("extract_keywords_keyness ranks by Log Ratio and filters by adjusted p", {
  skip_if_not_installed("quanteda.textstats")
  fx <- keyness_fixture()
  by_ratio <- extract_keywords_keyness(fx$dfm, target = fx$target, top_n = 50, rank_by = "log_ratio")
  expect_equal(abs(by_ratio$Log_Ratio), sort(abs(by_ratio$Log_Ratio), decreasing = TRUE))

  sig <- extract_keywords_keyness(fx$dfm, target = fx$target, top_n = 50, significant_only = TRUE)
  expect_true(all(sig$P_Adjusted < 0.05))
})

test_that("calculate_log_odds_ratio flags significance with Benjamini-Hochberg adjustment", {
  fx <- keyness_fixture()
  quanteda::docvars(fx$dfm, "grp") <- ifelse(fx$target, "A", "B")
  lo <- calculate_log_odds_ratio(fx$dfm, group_var = "grp", top_n = 100)
  expect_type(lo$significant, "logical")
  expect_lte(sum(lo$significant), sum(abs(lo$z_score) >= 1.96))
})

test_that("MATTR uses a fixed window that does not depend on other documents", {
  long_doc <- paste(rep(c("alpha", "beta", "gamma", "delta"), 40), collapse = " ")
  short_doc <- paste(letters[1:12], collapse = " ")
  mattr <- function(texts) {
    suppressMessages(lexical_diversity_analysis(quanteda::tokens(texts), measures = "MATTR"))$lexical_diversity$MATTR
  }

  alone <- mattr(long_doc)
  with_short <- mattr(c(long_doc, short_doc))
  with_empty <- mattr(c(long_doc, ""))

  expect_equal(alone, with_short[1])
  expect_equal(alone, with_empty[1])
  expect_true(is.na(with_short[2]))
  expect_false(any(is.infinite(with_empty)))
})

test_that("lexical diversity lowercases tokens and validates the window", {
  toks <- quanteda::tokens(paste(rep("The the THE cat", 20), collapse = " "))
  res <- suppressMessages(lexical_diversity_analysis(toks, measures = "MATTR"))
  expect_equal(res$lexical_diversity$MATTR, 2 / 50)
  expect_equal(res$summary_stats$window, 50L)
  expect_error(lexical_diversity_analysis(toks, measures = "MATTR", window = 1))
})

test_that("detect_language works with the NULL default", {
  skip_if_not_installed("stopwords")
  out <- detect_language(c("the cat sat on the mat and the dog ran", "this is a sentence in english"))
  expect_equal(out$language[1], "en")
})
