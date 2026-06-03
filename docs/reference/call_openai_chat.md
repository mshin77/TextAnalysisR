# Call OpenAI Chat Completion API

Makes a chat completion request to OpenAI's API.

## Usage

``` r
call_openai_chat(
  system_prompt,
  user_prompt,
  model = "gpt-4.1-mini",
  temperature = 0,
  max_tokens = 150,
  api_key
)
```

## Arguments

- system_prompt:

  Character string with system instructions

- user_prompt:

  Character string with user message

- model:

  Character string specifying the model (default: "gpt-4.1-mini")

- temperature:

  Numeric temperature for response randomness (default: 0)

- max_tokens:

  Maximum number of tokens to generate (default: 150)

- api_key:

  Character string with OpenAI API key

## Value

Character string with the model's response
