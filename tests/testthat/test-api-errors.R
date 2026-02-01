test_that(".parse_provider_error extracts OpenAI error message", {
  body <- '{"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error"}}'
  result <- TextAnalysisR:::.parse_provider_error("openai", body)
  expect_equal(result, "Incorrect API key provided.")
})

test_that(".parse_provider_error extracts Gemini error message", {
  body <- '{"error": {"message": "API key not valid.", "status": "INVALID_ARGUMENT"}}'
  result <- TextAnalysisR:::.parse_provider_error("gemini", body)
  expect_equal(result, "API key not valid.")
})

test_that(".parse_provider_error falls back to Gemini status field", {
  body <- '{"error": {"status": "PERMISSION_DENIED"}}'
  result <- TextAnalysisR:::.parse_provider_error("gemini", body)
  expect_equal(result, "PERMISSION_DENIED")
})

test_that(".parse_provider_error handles malformed JSON", {
  result <- TextAnalysisR:::.parse_provider_error("openai", "not json at all")
  expect_null(result)
})

test_that(".parse_provider_error handles empty body", {
  result <- TextAnalysisR:::.parse_provider_error("openai", "")
  expect_null(result)
})

test_that(".parse_provider_error truncates long messages", {
  long_msg <- paste(rep("x", 400), collapse = "")
  body <- sprintf('{"error": {"message": "%s"}}', long_msg)
  result <- TextAnalysisR:::.parse_provider_error("openai", body)
  expect_true(nchar(result) <= 300)
  expect_true(grepl("\\.\\.\\.$", result))
})

test_that(".format_api_error_message returns correct status meaning for 401", {
  msg <- TextAnalysisR:::.format_api_error_message("openai", "chat", 401, '{}')
  expect_match(msg, "Authentication failed")
  expect_match(msg, "OPENAI_API_KEY")
  expect_match(msg, "HTTP 401")
})

test_that(".format_api_error_message returns correct status meaning for 429", {
  msg <- TextAnalysisR:::.format_api_error_message("gemini", "chat", 429, '{}')
  expect_match(msg, "Rate limit exceeded")
  expect_match(msg, "Wait and retry")
})

test_that(".format_api_error_message returns correct status meaning for 500", {
  msg <- TextAnalysisR:::.format_api_error_message("openai", "embeddings", 500, '{}')
  expect_match(msg, "Server error")
  expect_match(msg, "Provider-side issue")
})

test_that(".format_api_error_message includes provider error detail", {
  body <- '{"error": {"message": "You exceeded your current quota."}}'
  msg <- TextAnalysisR:::.format_api_error_message("openai", "chat", 429, body)
  expect_match(msg, "Provider message: You exceeded your current quota.")
})

test_that(".format_api_error_message uses correct provider label", {
  msg_openai <- TextAnalysisR:::.format_api_error_message("openai", "chat", 401, '{}')
  msg_gemini <- TextAnalysisR:::.format_api_error_message("gemini", "chat", 401, '{}')
  expect_match(msg_openai, "^OpenAI")
  expect_match(msg_gemini, "^Gemini")
})

test_that(".stop_api_error raises error with formatted message", {
  expect_error(
    TextAnalysisR:::.stop_api_error("openai", "chat", 401, '{"error": {"message": "Bad key"}}'),
    "Authentication failed"
  )
  expect_error(
    TextAnalysisR:::.stop_api_error("openai", "chat", 401, '{"error": {"message": "Bad key"}}'),
    "Provider message: Bad key"
  )
})

test_that(".missing_api_key_message returns package format for openai", {
  msg <- TextAnalysisR:::.missing_api_key_message("openai", "package")
  expect_match(msg, "No OpenAI API key found")
  expect_match(msg, "OPENAI_API_KEY")
  expect_match(msg, "Sys.setenv")
  expect_match(msg, "\\.Renviron")
  expect_match(msg, "api_key parameter")
  expect_match(msg, "Ollama")
})

test_that(".missing_api_key_message returns package format for gemini", {
  msg <- TextAnalysisR:::.missing_api_key_message("gemini", "package")
  expect_match(msg, "No Gemini API key found")
  expect_match(msg, "GEMINI_API_KEY")
})

test_that(".missing_api_key_message returns shiny format for openai", {
  msg <- TextAnalysisR:::.missing_api_key_message("openai", "shiny")
  expect_match(msg, "OpenAI API key required")
  expect_match(msg, "API Key field")
  expect_match(msg, "OPENAI_API_KEY")
  expect_match(msg, "\\.Renviron")
})

test_that(".missing_api_key_message returns shiny format for gemini", {
  msg <- TextAnalysisR:::.missing_api_key_message("gemini", "shiny")
  expect_match(msg, "Gemini API key required")
  expect_match(msg, "GEMINI_API_KEY")
})
