# Cybersecurity Utility Functions

Functions for input validation, sanitization, and security logging

## Usage

``` r
validate_file_upload(file_info)
```

## Arguments

- file_info:

  File info object from Shiny fileInput

## Value

TRUE if valid, stops with error message if invalid

## NIST Compliance

This package follows NIST security standards (based on NIST SP 800-53):

- SC-8: Transmission Confidentiality and Integrity (HTTPS encryption)

- SC-28: Protection of Information at Rest (secure API key storage)

- IA-5: Authenticator Management (API key validation and format
  checking)

- AC-3: Access Enforcement (rate limiting, input validation, file type
  restrictions)

- SI-10: Information Input Validation (malicious content detection)

- AU-2: Audit Events (security logging and monitoring) Validate File
  Upload
