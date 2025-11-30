# Check Rate Limit

Check Rate Limit

## Usage

``` r
check_rate_limit(
  session_token,
  user_requests,
  max_requests = 100,
  window_seconds = 3600
)
```

## Arguments

- session_token:

  Shiny session token

- user_requests:

  Reactive value storing request history

- max_requests:

  Maximum requests allowed in time window

- window_seconds:

  Time window in seconds (default: 3600 = 1 hour)

## Value

TRUE if within limit, stops with error if exceeded
