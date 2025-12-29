test_that("package loads successfully", {
  expect_true(requireNamespace("TextAnalysisR", quietly = TRUE))
})

test_that("example dataset is accessible", {
  data(SpecialEduTech, package = "TextAnalysisR")
  expect_true(exists("SpecialEduTech"))
  expect_s3_class(SpecialEduTech, "data.frame")
  expect_true(nrow(SpecialEduTech) > 0)
})

test_that("unite_cols works", {
  skip_if_not_installed("quanteda")

  data(SpecialEduTech, package = "TextAnalysisR")
  test_data <- SpecialEduTech[1:3, c("title", "abstract")]

  united <- unite_cols(test_data, c("title", "abstract"))
  expect_true("united_texts" %in% names(united))
  expect_equal(nrow(united), 3)
})

test_that("prep_texts works", {
  skip_if_not_installed("quanteda")

  data(SpecialEduTech, package = "TextAnalysisR")
  test_data <- data.frame(
    united_texts = SpecialEduTech$abstract[1:3],
    stringsAsFactors = FALSE
  )

  tokens <- prep_texts(test_data, text_field = "united_texts")
  expect_true(inherits(tokens, "tokens"))
})

test_that("plot_word_frequency works", {
  skip_if_not_installed("quanteda")
  skip_if_not_installed("plotly")

  data(SpecialEduTech, package = "TextAnalysisR")
  texts <- SpecialEduTech$abstract[1:5]
  corp <- quanteda::corpus(texts)
  toks <- quanteda::tokens(corp)
  dfm_obj <- quanteda::dfm(toks)

  plot <- plot_word_frequency(dfm_obj, n = 5)
  expect_s3_class(plot, "plotly")
})

test_that("sentiment_lexicon_analysis works with basic texts", {
  skip_if_not_installed("quanteda")
  skip_if_not_installed("tidytext")

  data(SpecialEduTech, package = "TextAnalysisR")
  texts <- SpecialEduTech$abstract[1:5]

  corp <- quanteda::corpus(texts)
  toks <- quanteda::tokens(corp)
  dfm_obj <- quanteda::dfm(toks)

  result <- sentiment_lexicon_analysis(
    dfm_object = dfm_obj,
    lexicon = "bing",
    feature_type = "words"
  )

  expect_type(result, "list")
  expect_true("document_sentiment" %in% names(result))
  expect_true("summary_stats" %in% names(result))
  expect_equal(result$feature_type, "words")
})

test_that("calculate_text_readability works", {
  skip_if_not_installed("quanteda")
  skip_if_not_installed("quanteda.textstats")

  data(SpecialEduTech, package = "TextAnalysisR")
  texts <- SpecialEduTech$abstract[1:5]

  result <- calculate_text_readability(
    texts = texts,
    metrics = c("flesch", "gunning_fog")
  )

  expect_s3_class(result, "data.frame")
  expect_true("Document" %in% names(result))
  expect_equal(nrow(result), 5)
})

test_that("lexical_frequency_analysis works", {
  skip_if_not_installed("quanteda")
  skip_if_not_installed("plotly")

  data(SpecialEduTech, package = "TextAnalysisR")
  texts <- SpecialEduTech$abstract[1:5]

  corp <- quanteda::corpus(texts)
  toks <- quanteda::tokens(corp, remove_punct = TRUE)
  dfm_obj <- quanteda::dfm(toks)

  result <- lexical_frequency_analysis(dfm_obj, n = 5)

  expect_s3_class(result, "plotly")
})

test_that("extract_keywords_tfidf works", {
  skip_if_not_installed("quanteda")

  data(SpecialEduTech, package = "TextAnalysisR")
  texts <- SpecialEduTech$abstract[1:5]

  corp <- quanteda::corpus(texts)
  toks <- quanteda::tokens(corp)
  dfm_obj <- quanteda::dfm(toks)

  result <- extract_keywords_tfidf(dfm_obj, top_n = 3)

  expect_s3_class(result, "data.frame")
  expect_true("Keyword" %in% names(result))
  expect_true("TF_IDF_Score" %in% names(result))
  expect_true("Frequency" %in% names(result))
})

test_that("run_app function exists", {
  expect_true(exists("run_app"))
  expect_type(run_app, "closure")
})
