test_that(".hpd is the shortest interval covering the mass", {
  set.seed(1)
  x <- stats::rexp(4000)
  h <- TextAnalysisR:::.hpd(x, 0.9)
  e <- as.numeric(stats::quantile(x, c(0.05, 0.95)))
  expect_lte(h[1], h[2])
  expect_lte(h[2] - h[1], (e[2] - e[1]) + 1e-8)
  expect_gte(mean(x >= h[1] & x <= h[2]), 0.88)
})

test_that(".effect_interval dispatches eti vs hpd", {
  set.seed(1)
  x <- stats::rnorm(2000)
  expect_equal(TextAnalysisR:::.effect_interval(x, "eti", 0.95),
               as.numeric(stats::quantile(x, c(0.025, 0.975), names = FALSE)))
  expect_equal(TextAnalysisR:::.effect_interval(x, "hpd", 0.95),
               TextAnalysisR:::.hpd(x, 0.95))
})

test_that("estimate_topic_effects: stm parity, hpd option, and bounded beta", {
  skip_on_cran()
  skip_if_not_installed("quanteda")
  skip_if_not_installed("stm")
  skip_if_not_installed("betareg")
  suppressPackageStartupMessages({ library(quanteda); library(stm) })

  data(SpecialEduTech, package = "TextAnalysisR")
  d <- SpecialEduTech
  d$text <- paste(d$title, d$abstract)
  toks <- tokens(corpus(d, text_field = "text"), remove_punct = TRUE, remove_numbers = TRUE)
  toks <- tokens_remove(tokens_tolower(toks), stopwords("en"))
  dfmat <- dfm_trim(dfm(toks), min_termfreq = 5, min_docfreq = 3)
  dfmat <- dfm_subset(dfmat, ntoken(dfmat) > 0)
  o <- convert(dfmat, to = "stm")
  set.seed(1)
  m <- stm(o$documents, o$vocab, K = 3, prevalence = ~ reference_type,
           data = o$meta, max.em.its = 10, init.type = "Spectral", verbose = FALSE)
  e <- estimateEffect(1:3 ~ reference_type, m, metadata = o$meta, uncertainty = "Global")

  s_eti <- estimate_topic_effects(e, "reference_type", "pointestimate")
  expect_named(s_eti, c("topic", "value", "proportion", "lower", "upper"))
  expect_true(all(s_eti$lower <= s_eti$proportion & s_eti$proportion <= s_eti$upper))

  set.seed(2); s1 <- estimate_topic_effects(e, "reference_type", "pointestimate", interval = "eti")
  set.seed(2); s2 <- estimate_topic_effects(e, "reference_type", "pointestimate", interval = "hpd")
  expect_equal(s1$proportion, s2$proportion)

  b <- estimate_topic_effects(e, "reference_type", "pointestimate",
                              method = "beta", model = m, documents = o$documents, nsims = 8)
  expect_true(all(b$proportion > 0 & b$proportion < 1))
  expect_true(all(b$lower > 0 & b$upper < 1))

  expect_error(estimate_topic_effects(e, "reference_type", "pointestimate", method = "beta"),
               "needs")
})
