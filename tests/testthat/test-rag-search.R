test_that("run_rag_search validates document input", {
  skip_if_not_installed("reticulate")

  env_check <- tryCatch(
    check_python_env(),
    error = function(e) list(available = FALSE)
  )

  skip_if(!env_check$available, "Python environment not available")

  result <- run_rag_search(
    query = "Test question?",
    documents = character(0)
  )

  expect_false(result$success)
  expect_match(result$error, "No documents provided")
})

test_that("run_rag_search requires Python environment", {
  skip_if_not_installed("reticulate")

  env_check <- tryCatch(
    check_python_env(),
    error = function(e) list(available = FALSE)
  )

  skip_if(!env_check$available, "Python environment not available")

  result <- run_rag_search(
    query = "What is assistive technology?",
    documents = c("Assistive tech helps students", "Technology supports learning")
  )

  expect_type(result, "list")
  expect_true("success" %in% names(result))

  if (!result$success) {
    expect_match(result$error, "LangGraph|Python|ModuleNotFoundError|Error in RAG search")
  }
})

test_that("run_rag_search returns expected structure", {
  skip_if_not_installed("reticulate")

  env_check <- tryCatch(
    check_python_env(),
    error = function(e) list(available = FALSE)
  )

  skip_if(!env_check$available, "Python environment not available")

  result <- run_rag_search(
    query = "What is special education?",
    documents = c(
      "Special education provides support for students with disabilities.",
      "Individualized Education Programs guide special education services.",
      "Assistive technology is used in special education."
    ),
    ollama_model = "llama3",
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
