# StayReady Security Notes

## Current Posture

StayReady is a public Flask application with no accounts, payments, uploads, admin
dashboard, or user-owned database records. This file documents controls that were
reviewed; it is not a claim that the application is secure or vulnerability-free.

The app validates public inputs, rate-limits expensive geocoding/map endpoints,
uses allowlisted outbound service hosts, applies baseline browser security
headers, and returns generic public errors. Official-data failures must remain
unknown, unavailable, or not checked and must never become reassuring results.

## Data Handling

StayReady currently stores official public source data in repository files and
may read public official-data tables from Supabase. It does not intentionally
write user addresses, email addresses, household profiles, medical details, or
access needs to Supabase or another durable application database.

Address and optional household selections are held temporarily in the signed
Flask session so the result pages work. They are not logged intentionally. The
session cookie is HttpOnly, SameSite=Lax, and Secure in production. Flask signed
cookies provide integrity, not confidentiality, so sensitive free-text medical
details should not be requested or added without a separate privacy design.

## Secrets

Local development uses an untracked `.env`. Render must define a random
`FLASK_SECRET_KEY` of at least 32 characters. Only the public Supabase anon key
is supported. Never add a service-role key to this app, browser JavaScript,
repository files, logs, or health responses.

## Rate Limits

- Address lookup: 20 requests per minute per apparent client IP.
- Flood map filtering: 20 requests per minute per apparent client IP.
- CGS remote map layers: 20 requests per minute per apparent client IP.
- USGS live earthquake feed: 20 requests per minute per apparent client IP.

The built-in limiter is process-local. Before material public traffic, replace
it with a shared Cloudflare or Redis-backed limit so all Gunicorn workers share
the same counters.

## Supabase

Supabase is optional and read-only in Phase 1. RLS must remain enabled. The anon
role may select public official-source tables but must not insert, update, or
delete. Any future feedback or user-generated table requires separate RLS,
retention rules, and privacy review before launch.

## Vulnerability Reports

Do not include real addresses, credentials, medical details, or exploit data in
a public issue. Send a private report to the repository owner with the affected
route, reproduction steps using fake data, and expected impact. Add a dedicated
security contact before public launch.

## Pre-Launch Checklist

- Rotate any credential that was ever pasted into chat, screenshots, or commits.
- Enable GitHub secret scanning, Dependabot alerts, and branch protection.
- Confirm Render uses HTTPS, `FLASK_ENV=production`, and a generated secret.
- Confirm Render logs do not include request bodies or full query strings.
- Put Cloudflare rate limiting/WAF rules in front of geocode and map APIs.
- Review Cloudflare proxy-header handling before trusting client IP limits.
- Re-run `pip-audit`, tests, and a manual browser check after dependency updates.
- Verify Supabase RLS and anon policies directly in the dashboard.
- Enable Supabase backups or point-in-time recovery appropriate to the plan.
- Configure uptime/error monitoring without collecting full addresses.
- Review CSP deployment separately; current CDN and inline template assets need
  nonce/hash work before a strict policy can be enabled safely.
- Document backup/restore for official source records and dataset provenance.
- Confirm all unavailable data still fails as unknown/unavailable, never safe.

## Deferred Items

A strict Content-Security-Policy, distributed rate limiting, automated security
alerts, restore drills, and formal penetration testing require deployment
configuration or external services. Authentication, authorization, tenant
isolation, password reset, JWT, webhooks, and payments are not applicable until
those features exist and must receive a new threat review before implementation.
