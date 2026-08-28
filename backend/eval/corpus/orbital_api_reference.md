# Orbital Platform API — Reference v3.2

## Authentication

The Orbital API authenticates requests using a bearer token supplied in the
`Authorization` header. Tokens are issued from the developer console and are
scoped to a single workspace.

Access tokens expire after 3600 seconds. Refresh tokens are valid for ninety
days from issuance and are single-use: each refresh returns a new refresh token
and invalidates the one presented. Attempting to reuse a consumed refresh token
returns error code E-4021 and revokes the entire token family as a precaution
against replay attacks.

Service accounts authenticate with a signed JWT assertion rather than a static
token. The assertion must be signed with RS256 and must include an `aud` claim
set to `https://api.orbital.example/token`.

## Rate Limits

Rate limits are applied per workspace, not per token. The default sustained
limit is 600 requests per minute, with a burst allowance of 100 requests
within any rolling ten-second window.

Exceeding the sustained limit returns HTTP 429 with error code E-4290 and a
`Retry-After` header expressed in seconds. Exceeding the burst allowance
returns error code E-4291. Clients should implement exponential backoff with
full jitter; retrying without backoff after a 429 will extend the cooldown
period rather than shorten it.

Batch endpoints are metered differently: a single batch call counts against the
limit as one request per twenty contained operations, rounded up.

## Error Codes

| Code    | HTTP | Meaning                                              |
|---------|------|------------------------------------------------------|
| E-4010  | 401  | Missing or malformed Authorization header            |
| E-4011  | 401  | Access token expired                                 |
| E-4021  | 401  | Refresh token already consumed; token family revoked |
| E-4030  | 403  | Token scope insufficient for the requested resource  |
| E-4041  | 404  | Resource does not exist in this workspace            |
| E-4090  | 409  | Idempotency key reused with a different request body |
| E-4220  | 422  | Request body failed schema validation                |
| E-4290  | 429  | Sustained rate limit exceeded                        |
| E-4291  | 429  | Burst rate limit exceeded                            |
| E-5030  | 503  | Upstream dependency unavailable; retry is safe       |

## Pagination

List endpoints return a maximum of 100 items per page. The default page size is
25. Pagination is cursor-based: responses include a `next_cursor` field, which
is null when the final page has been reached. Offset pagination is not
supported, because it produces inconsistent results when the underlying
collection is mutated between requests.

Cursors are opaque and expire after fifteen minutes. A request presenting an
expired cursor returns error code E-4220 and must restart pagination from the
beginning.

## Idempotency

All mutating requests accept an `Idempotency-Key` header. Keys are retained for
twenty-four hours. Replaying a request with the same key and an identical body
returns the original response without re-executing the operation. Replaying a
key with a different body returns error code E-4090.

Idempotency keys must be unique per workspace and should be generated as UUID
version 4 values. Keys longer than 255 characters are rejected.

## Webhooks

Webhook deliveries are signed using HMAC-SHA256. The signature is transmitted
in the `Orbital-Signature` header, formatted as `t=<timestamp>,v1=<signature>`.
Consumers must compute the expected signature over the concatenation of the
timestamp, a period character, and the raw request body.

To defend against replay attacks, reject any delivery whose timestamp is more
than five minutes old. Orbital retries failed deliveries up to eight times with
exponential backoff over a total window of twenty-four hours. A delivery is
considered failed if the endpoint does not return a 2xx status within ten
seconds.

Webhook endpoints must be registered with an HTTPS URL. Plain HTTP endpoints
are rejected at registration time.
