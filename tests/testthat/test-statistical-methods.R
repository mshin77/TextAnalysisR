# regression tests for hand-implemented statistics

test_that("c-TF-IDF matches the class-based formula", {
  texts <- c("apple banana apple", "banana cherry", "dog cat dog", "cat mouse")
  assignments <- c(1, 1, 2, 2)

  kw <- TextAnalysisR:::generate_semantic_topic_keywords(
    texts, assignments, n_keywords = 3, method = "c-tfidf"
  )

  expect_length(kw, 2)
  expect_true(all(lengths(kw) > 0))
  expect_true("apple" %in% kw[["1"]])
  expect_true("dog" %in% kw[["2"]])
})

test_that("mmr diversity changes keyword selection", {
  set.seed(7)
  texts <- TextAnalysisR::SpecialEduTech$abstract[1:40]
  assignments <- rep(1:2, each = 20)

  kw_low <- TextAnalysisR:::generate_semantic_topic_keywords(
    texts, assignments, n_keywords = 8, method = "mmr", diversity = 0.1
  )
  kw_high <- TextAnalysisR:::generate_semantic_topic_keywords(
    texts, assignments, n_keywords = 8, method = "mmr", diversity = 0.9
  )

  expect_true(all(lengths(kw_low) == 8))
  expect_false(identical(kw_low, kw_high))
})

test_that("keyword stability matches topics across permuted labels", {
  kws <- list(
    a = c("math", "algebra", "geometry"),
    b = c("reading", "phonics", "fluency"),
    c = c("autism", "behavior", "social")
  )

  expect_equal(calculate_keyword_stability(kws, kws[c(3, 1, 2)]), 1)
  expect_equal(
    calculate_keyword_stability(
      kws,
      list(x = c("zebra", "lion"), y = c("piano", "violin"), z = c("soccer", "tennis"))
    ),
    0
  )
})

test_that("weighted log odds uses full-vocabulary totals", {
  skip_if_not_installed("tidylo")

  abstracts <- TextAnalysisR::SpecialEduTech[1:40, ]
  corp <- quanteda::corpus(
    abstracts$abstract,
    docvars = data.frame(reference_type = abstracts$reference_type)
  )
  dfm_object <- quanteda::dfm(quanteda::tokens(corp, remove_punct = TRUE))

  result <- calculate_weighted_log_odds(dfm_object, "reference_type",
                                        top_n = 5, min_count = 2)

  expect_true(all(result$n >= 2))
  expect_true(all(is.finite(result$log_odds_weighted)))
  expect_type(result$significant, "logical")

  # z-scores must match tidylo computed on the unfiltered complete grid
  tidy_full <- tidytext::tidy(dfm_object) %>%
    dplyr::left_join(
      data.frame(document = quanteda::docnames(dfm_object),
                 reference_type = abstracts$reference_type),
      by = "document"
    ) %>%
    dplyr::group_by(reference_type, term) %>%
    dplyr::summarise(n = sum(count), .groups = "drop") %>%
    tidyr::complete(reference_type, term, fill = list(n = 0))
  expected <- tidylo::bind_log_odds(tidy_full, set = reference_type,
                                    feature = term, n = n)
  merged <- merge(result, expected,
                  by.x = c("reference_type", "feature"),
                  by.y = c("reference_type", "term"))
  expect_equal(merged$log_odds_weighted.x, merged$log_odds_weighted.y,
               tolerance = 1e-10)
})

test_that("log odds ratio returns finite Monroe-style estimates", {
  abstracts <- TextAnalysisR::SpecialEduTech[1:40, ]
  corp <- quanteda::corpus(
    abstracts$abstract,
    docvars = data.frame(reference_type = abstracts$reference_type)
  )
  dfm_object <- quanteda::dfm(quanteda::tokens(corp, remove_punct = TRUE))

  result <- calculate_log_odds_ratio(dfm_object, "reference_type", top_n = 5)

  expect_true(all(result$odds1 > 0))
  expect_true(all(result$odds2 > 0))
  expect_true(all(is.finite(result$z_score)))
  expect_type(result$significant, "logical")
})

test_that("keyness handles every documented measure", {
  abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:30]
  dfm_object <- quanteda::dfm(quanteda::tokens(quanteda::corpus(abstracts),
                                               remove_punct = TRUE))

  for (m in c("lr", "chi2", "exact", "pmi")) {
    kw <- extract_keywords_keyness(dfm_object, target = 1:5, top_n = 5, measure = m)
    expect_true(is.numeric(kw$Keyness_Score), info = m)
    expect_equal(nrow(kw), 5, info = m)
  }
  expect_error(extract_keywords_keyness(dfm_object, target = 1, measure = "bogus"))
})

test_that("dispersion metrics stay in bounds", {
  toks <- quanteda::tokens(quanteda::corpus(TextAnalysisR::SpecialEduTech$abstract[1:30]))
  disp <- calculate_dispersion_metrics(toks, c("students", "the"))

  expect_true(all(disp$rosengren_s <= 1 + 1e-9, na.rm = TRUE))
  expect_true(all(disp$rosengren_s > 0, na.rm = TRUE))
  expect_true(all(disp$juilland_d >= 0 & disp$juilland_d <= 1, na.rm = TRUE))
})

test_that("mtld skips short factors and hdd stays in bounds", {
  toks <- rep(c("the", "the", "cat", "sat", "on", "a", "mat", "with",
                "her", "big", "hat", "today"), 12)

  expect_true(is.finite(TextAnalysisR:::.calc_mtld(toks)))
  expect_true(TextAnalysisR:::.calc_mtld(toks) > 0)

  hdd <- TextAnalysisR:::.calc_hdd(toks)
  expect_true(hdd > 0 && hdd <= 1)
  expect_true(is.na(TextAnalysisR:::.calc_hdd(toks[1:10])))
})
