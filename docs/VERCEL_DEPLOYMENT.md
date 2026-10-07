# Deploy SmartLook to Vercel

## Deployment configuration

- Framework: **Flask**
- Root directory: repository root
- Entry point: `app.py`, exported `app`
- Python: **3.12**, pinned in `.python-version`
- Build command / output directory: leave unset (Vercel Flask defaults)
- Function duration: **300 seconds**, configured on `app.py` in `vercel.json`
- Static files: `public/assets/`, served at `/assets/`
- The Procfile is only for other Gunicorn-based hosts; Vercel does not use it.

## Required settings

Set these in the project's Settings → Environment Variables. Use separate
values/databases for Preview and Production wherever possible. Never paste
credentials into a Git commit, issue, PR, or a browser-side JavaScript file.

| Variable | Value / source |
| --- | --- |
| `GROQ_API_KEY` | API key from your Groq account |
| `GOOGLE_CLOUD_PROJECT` | Google project in which query jobs run and are billed |
| `GCP_SERVICE_ACCOUNT_JSON` | Complete JSON service-account credential, as a single environment value (`GCP_SERVICE_ACCOUNT` is also accepted) |
| `APP_SECRET_KEY` | At least 32 random characters; generate locally with `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `APP_ACCESS_PASSWORD` | A strong password of at least 12 characters, shared only with intended demo users |
| `UPSTASH_REDIS_REST_URL` | HTTPS endpoint from an Upstash Redis database |
| `UPSTASH_REDIS_REST_TOKEN` | Read/write REST token for that Redis database |

Vercel Marketplace integrations may provide `KV_REST_API_URL` and
`KV_REST_API_TOKEN` instead. These aliases are supported. Do not use a read-only
token: history, locks, and rate counters require writes.

Recommended setup: create/connect Upstash Redis from the Vercel Marketplace,
choose a plan intentionally, and scope its injected variables to the intended
environment. No database is provisioned automatically by application code.

Google setup: enable the BigQuery API and give the service account permission to
create query jobs in `GOOGLE_CLOUD_PROJECT` (typically BigQuery Job User). The
queries read `bigquery-public-data.thelook_ecommerce`; do not grant project
Editor or Owner just to run this demo. Confirm the project's billing/quota setup
supports these queries. The app's SQL allowlist is defense in depth, not a
replacement for Google IAM.

`GROQ_MODEL` defaults to `openai/gpt-oss-120b`; verify that your Groq account has
access. The application does not use the README's former
`GOOGLE_APPLICATION_CREDENTIALS` / `BIGQUERY_PROJECT_ID` variables.

## Default limits

| Setting | Default |
| --- | --- |
| `CHAT_REQUESTS_PER_MINUTE` | 5 per session |
| `IP_REQUESTS_PER_MINUTE` | 10 per IP |
| `DAILY_CHAT_LIMIT` | 200 per environment, UTC day |
| Login attempts | 10 per IP per 15-minute window; 600 globally per hour |
| `BIGQUERY_MAX_BYTES_BILLED` | 1,000,000,000 bytes per query |
| `BIGQUERY_TIMEOUT_SECONDS` | 45 seconds maximum; cancellation attempted on timeout |
| `MAX_RESULT_ROWS` | 1,000 maximum; ordinary generated queries default to 100 |
| `GROQ_TIMEOUT_SECONDS` | 30 seconds per request, retries disabled |
| `GROQ_MAX_TOKENS` | 4,096 output tokens per model call |
| `SESSION_TTL_SECONDS` | 86,400 (24 hours) |
| Message / response bounds | 4,000 input characters / 3.5 MB encoded JSON |

Rate limits use Redis atomic operations and fail closed if storage is unavailable.
Counters include attempted chat requests, including provider failures. The daily
cap limits requests, not currency spend. BigQuery and Groq may have their own
billing and quotas. Long or broad requests should be narrowed rather than
raising limits without checking costs.

## Release procedure

1. Import the repository/branch into the `smartlook-ai-agent` project, or link
   the reviewed local checkout to that exact project and team.
2. Add the required environment values; connect Redis.
3. Deploy a Preview and confirm the build succeeds. Until required settings are
   present, the app returns a setup page/503 instead of exposing unprotected AI.
4. Open the preview in an authenticated browser if Vercel protection applies.
5. Sign in with the app access password. Verify a greeting, a read-only data
   question, a follow-up, a chart, CSV and PNG downloads, and a cohort question.
6. Use a second browser profile to confirm histories remain separate. Reload
   the first tab and switch conversations to check context persistence.
7. Check `/api/health`: `ready` means required configuration is present; it does
   **not** prove Groq, BigQuery, or Redis credentials work. Live chat is the
   credential/integration check. Health checks never make paid provider calls.
8. For production, stage with `vercel deploy --prod --skip-domain`, test that
   exact production build, then `vercel promote <deployment-url>`.

Environment changes require redeployment. Do not promote the setup-only build
as a functioning AI service.

## Verification boundaries

Automated tests use provider stubs and a Redis emulator to validate real route,
agent graph, SQL restriction, concurrency, and Redis Lua behavior. Browser tests
use a deterministic agent through the real Flask routes. These checks do not
replace testing actual Groq, Google Cloud, and Upstash credentials on Vercel.

Linux x86_64 Python 3.12 dependency wheels were resolved and installed locally
for packaging inspection. The dependency files total roughly 249 MiB before
Vercel bytecode/packaging overhead; the normal Python bundle limit is 500 MB.
A successful cloud build remains the definitive deployment check.

## Troubleshooting

- **Setup page / 503:** verify the secret key, access password length, and both
  Redis REST values; for chat also verify all three AI credential variables.
- **401:** sign in again; rotating `APP_SECRET_KEY` invalidates existing cookies.
- **409:** a request from the same browser session is already running. Wait; a
  worker interrupted by the platform releases its lock by expiry after 350s.
- **429:** a configured request budget is exhausted. The generic Retry-After is
  60s; daily budgets reset at the next UTC day and login budgets after 15 minutes.
- **413:** narrow the question/result. Data is returned only once in responses.
- **Storage unavailable:** verify Redis token permissions, connectivity, and
  quotas. At 30 saved conversations, clear history before starting more chats.
- **AI/query error:** verify model access, BigQuery IAM/billing, and query limits.
  Keep credentials out of logs when investigating provider errors.

Official references: [Flask on Vercel](https://vercel.com/docs/frameworks/backend/flask),
[Python runtime](https://vercel.com/docs/functions/runtimes/python),
[Function limits](https://vercel.com/docs/functions/limitations),
[Upstash REST API](https://upstash.com/docs/redis/features/restapi).
