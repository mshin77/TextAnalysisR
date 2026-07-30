## Input Validation

- File uploads: Extension whitelist, 100MB limit (50MB paste), malicious content scanning
- Text and LLM inputs: XSS and prompt injection filtering
- Column names: Regex validation to prevent formula injection

## API Key Security

- Stored via `.env` or environment variables (never logged or persisted)
- Masked input, format validation, transmitted via secure headers only

## Network Security

- Content Security Policy restricting script, style, and frame sources; clickjacking blocked via frame-ancestors
- HTTP Strict Transport Security (HSTS), MIME-sniffing protection (nosniff), and referrer-policy restriction
- HTTPS with TLS 1.2+ via Caddy/Cloudflare

## Data Protection

- Session-scoped with no persistent storage, cookies, or identifiers
- Rate limiting: 100 uploads/hour, 20 AI requests/hour per session
- Security event logging with sanitized error messages
- Local processing option (FERPA/HIPAA compatible)

## Infrastructure

- Cloudflare DNS with DDoS protection
- Docker + Caddy deployment
