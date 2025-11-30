# Security

TextAnalysisR includes built-in security features.

## Data Protection

| Feature            | Description                         |
|--------------------|-------------------------------------|
| File validation    | Only CSV, XLSX, TXT, RDS (max 50MB) |
| Input sanitization | Removes harmful code patterns       |
| API key validation | Format checking before use          |
| Rate limiting      | 100 requests/hour                   |

## API Key Storage

**In App:** Secure password field, session-only storage

**Environment Variable:**

``` r
Sys.setenv(OPENAI_API_KEY = "sk-...")
```

**Encrypted (Advanced):**

``` r
keyring::key_set("openai_api")
```

## Privacy

- 100% local processing with R package
- Optional local AI via Ollama
- HIPAA/FERPA compliant
- Works offline
