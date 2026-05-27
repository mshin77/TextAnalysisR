# Call Gemini Chat API

Makes a chat completion request to Google's Gemini API.

## Usage

``` r
call_gemini_chat(
  system_prompt,
  user_prompt,
  model = "gemini-2.5-flash",
  temperature = 0,
  max_tokens = 8192,
  api_key
)
```

## Arguments

- system_prompt:

  Character string with system instructions

- user_prompt:

  Character string with user message

- model:

  Character string specifying the Gemini model (default:
  "gemini-2.5-flash")

- temperature:

  Numeric temperature for response randomness (default: 0)

- max_tokens:

  Maximum number of tokens to generate (default: 150)

- api_key:

  Character string with Gemini API key

## Value

Character string with the model's response
