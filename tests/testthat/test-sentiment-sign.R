test_that(".sentiment_sign maps every label scheme to a consistent sign", {
  s <- TextAnalysisR:::.sentiment_sign
  expect_equal(s("positive"), 1)
  expect_equal(s("POSITIVE"), 1)
  expect_equal(s("negative"), -1)
  expect_equal(s("neutral"), 0)
  expect_equal(s("1 star"), -1)
  expect_equal(s("3 stars"), 0)
  expect_equal(s("5 stars"), 1)
  expect_true(is.na(s("banana")))
})

test_that("neural sentiment score and label stay sign-consistent (regression: neutral/star bug)", {
  labels <- c("positive", "neutral", "negative", "1 star", "5 stars")
  conf <- c(0.9, 0.8, 0.7, 0.95, 0.85)
  signs <- vapply(labels, TextAnalysisR:::.sentiment_sign, numeric(1))
  score <- signs * conf
  sentiment <- ifelse(is.na(signs) | signs == 0, "neutral",
                      ifelse(signs > 0, "positive", "negative"))
  expect_equal(unname(sentiment), c("positive", "neutral", "negative", "negative", "positive"))
  expect_true(all((score > 0) == (sentiment == "positive")))
  expect_true(all((score < 0) == (sentiment == "negative")))
  expect_equal(unname(score[sentiment == "neutral"]), 0)
})
