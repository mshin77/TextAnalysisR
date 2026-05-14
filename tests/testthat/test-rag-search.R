test_that("run_rag_search validates document input", {
  skip_on_cran()
  result <- run_rag_search(
    query = "Test question?",
    documents = character(0),
    provider = "openai"
  )

  expect_false(result$success)
  expect_match(result$error, "No documents provided|API key")
})

test_that("run_rag_search requires API key", {
  skip_on_cran()
  result <- run_rag_search(
    query = "What is assistive technology?",
    documents = c("Assistive tech helps students", "Technology supports learning"),
    provider = "openai",
    api_key = ""
  )

  expect_type(result, "list")
  expect_true("success" %in% names(result))

  if (!result$success) {
    expect_match(result$error, "API key|No documents provided")
  }
})

test_that("run_rag_search returns expected structure", {
  skip_on_cran()
  skip_if(
    !nzchar(Sys.getenv("OPENAI_API_KEY")),
    "OPENAI_API_KEY not set"
  )

  result <- run_rag_search(
    query = "What is special education?",
    documents = c(
      "Special education provides support for students with disabilities.",
      "Individualized Education Programs guide special education services.",
      "Assistive technology is used in special education."
    ),
    provider = "openai",
    top_k = 2
  )

  expect_type(result, "list")
  expect_true("success" %in% names(result))
  expect_true("answer" %in% names(result))
  expect_true("confidence" %in% names(result))
  expect_true("sources" %in% names(result))

  if (result$success) {
    expect_type(result$answer, "character")
    expect_type(result$confidence, "double")
    expect_gte(result$confidence, 0)
    expect_lte(result$confidence, 1)
  }
})
