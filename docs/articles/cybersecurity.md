# Security

``` r

library(TextAnalysisR)

mydata <- SpecialEduTech[seq_len(5), c("title", "abstract")]
united <- unite_cols(mydata, listed_vars = c("title", "abstract"))
toks   <- prep_texts(united, text_field = "united_texts")
quanteda::ndoc(toks)
```

    ## [1] 5

TextAnalysisR includes built-in security features.

## Input Validation

| Feature | Description |
|----|----|
| File uploads | Extension whitelist, 50MB limit, malicious content scanning |
| Text and LLM inputs | XSS and prompt injection filtering |
| Column names | Regex validation to prevent formula injection |

## API Key Security

- Stored via `.env` or environment variables (never logged or persisted)
- Masked input field with format validation
- Transmitted via secure headers only

**Environment Variable:** add `OPENAI_API_KEY=...` to `.Renviron`, or
set in-session:

``` r

Sys.setenv(OPENAI_API_KEY = "sk-...")
```

## Network Security

- Content Security Policy, X-Frame-Options, SRI for CDN resources
- HTTPS with TLS 1.2+ via Nginx/Cloudflare

## Data Protection

- Session-scoped with no persistent storage, cookies, or identifiers
- Rate limiting: 100 requests/hour per session
- Security event logging with sanitized error messages
- Local processing option (FERPA/HIPAA compatible)

## Infrastructure

- Cloudflare DNS with DDoS protection
- Docker + Nginx deployment
