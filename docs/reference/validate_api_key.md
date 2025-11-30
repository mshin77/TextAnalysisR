# Validate OpenAI API Key Format

Validates OpenAI API key format according to NIST IA-5(1) authenticator
management. Checks key prefix, length, and basic format requirements.

## Usage

``` r
validate_api_key(api_key, strict = TRUE)
```

## Arguments

- api_key:

  Character string containing the API key

- strict:

  Logical, if TRUE performs additional validation checks

## Value

Logical TRUE if valid, FALSE with warnings if invalid

## NIST Compliance

Implements NIST IA-5(1): Authenticator Management - Password-Based
Authentication. Validates format, length, and character composition to
prevent weak or malformed keys.

## Examples

``` r
if (FALSE) { # \dontrun{
validate_api_key("sk-proj...")
} # }
```
