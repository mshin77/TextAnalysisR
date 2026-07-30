test_that(".build_covariate_formula builds a resolvable RHS", {
  f <- TextAnalysisR:::.build_covariate_formula("reference_type + s(year, df = 4)")
  expect_s3_class(f, "formula")
  expect_equal(paste(deparse(f), collapse = " "), "~reference_type + s(year, df = 4)")
})

test_that(".validate_custom_formula accepts standard fixed-effects terms", {
  vars <- c("reference_type", "year", "period")
  valid <- c(
    "reference_type + s(year, df = 4)",
    "reference_type * year",
    "poly(year, 2)",
    "ns(year, df = 3)",
    "I((year >= 2000)*(year - 2000)) + I((year >= 2010)*(year - 2010))",
    "~ reference_type + year"
  )
  for (rhs in valid) {
    expect_null(TextAnalysisR:::.validate_custom_formula(rhs, vars), info = rhs)
  }
})

test_that(".validate_custom_formula rejects unknown vars and unsafe calls", {
  vars <- c("reference_type", "year")
  expect_type(TextAnalysisR:::.validate_custom_formula("", vars), "character")
  expect_type(TextAnalysisR:::.validate_custom_formula("nope + year", vars), "character")
  expect_type(TextAnalysisR:::.validate_custom_formula("system('x')", vars), "character")
  expect_type(TextAnalysisR:::.validate_custom_formula("base::mean(year)", vars), "character")
  expect_type(TextAnalysisR:::.validate_custom_formula("eval(parse(text = '1'))", vars), "character")
})
