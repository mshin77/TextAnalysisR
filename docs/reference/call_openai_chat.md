# Call OpenAI Chat Completion API

Internal function to call OpenAI's chat completion API.

## Usage

``` r
call_openai_chat(
  system_prompt,
  user_prompt,
  model = "gpt-3.5-turbo",
  temperature = 0,
  max_tokens = 150,
  api_key
)
```

## Arguments

- system_prompt:

  System message for the chat.

- user_prompt:

  User message/query.

- model:

  Model to use (default: "gpt-3.5-turbo").

- temperature:

  Sampling temperature (default: 0).

- max_tokens:

  Maximum tokens in response (default: 150).

- api_key:

  OpenAI API key.

## Value

Character string with the model's response.
