create table if not exists public.email_interests (
  id uuid primary key default gen_random_uuid(),
  email text not null,
  location text,
  user_type text,
  consent_version text not null,
  consented_at timestamptz not null,
  subscription_status text not null default 'subscribed',
  source text not null default 'homepage',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint email_interests_email_unique unique (email),
  constraint email_interests_email_normalized check (email = lower(btrim(email))),
  constraint email_interests_user_type_check check (
    user_type is null or user_type in (
      'resident',
      'parent',
      'student',
      'educator',
      'nonprofit_community',
      'cert_volunteer',
      'other'
    )
  ),
  constraint email_interests_subscription_status_check check (
    subscription_status in ('subscribed', 'unsubscribed', 'deleted')
  ),
  constraint email_interests_source_check check (
    source in ('homepage', 'feedback', 'manual_import')
  )
);

create table if not exists public.feedback_submissions (
  id uuid primary key default gen_random_uuid(),
  submission_category text not null,
  admin_tag text not null,
  name text,
  email text,
  organization text,
  role text,
  interest_type text,
  message text not null,
  page_context text not null,
  review_status text not null default 'new',
  retention_expires_at timestamptz not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint feedback_submissions_category_check check (
    submission_category in (
      'general_feedback',
      'confusing_language',
      'incorrect_source',
      'organization_demo',
      'partnership_sponsorship',
      'other_question',
      'community_interest'
    )
  ),
  constraint feedback_submissions_admin_tag_check check (
    admin_tag in (
      'resident_feedback',
      'bug_source_report',
      'organization_lead',
      'advisor_lead',
      'partnership',
      'spam'
    )
  ),
  constraint feedback_submissions_interest_type_check check (
    interest_type is null or interest_type in (
      'feedback_call',
      'demo',
      'pilot',
      'sponsorship',
      'partnership'
    )
  ),
  constraint feedback_submissions_page_context_check check (
    page_context in (
      'feedback',
      'home',
      'map',
      'privacy',
      'risk_summary',
      'sources',
      'terms'
    )
  ),
  constraint feedback_submissions_review_status_check check (
    review_status in ('new', 'reviewing', 'follow_up', 'closed', 'spam')
  )
);

create index if not exists idx_email_interests_status
on public.email_interests(subscription_status);

create index if not exists idx_email_interests_user_type
on public.email_interests(user_type);

create index if not exists idx_feedback_submissions_review_status
on public.feedback_submissions(review_status);

create index if not exists idx_feedback_submissions_admin_tag
on public.feedback_submissions(admin_tag);

create index if not exists idx_feedback_submissions_retention
on public.feedback_submissions(retention_expires_at);

alter table public.email_interests enable row level security;
alter table public.feedback_submissions enable row level security;

revoke all on table public.email_interests from anon, authenticated;
revoke all on table public.feedback_submissions from anon, authenticated;

grant select, insert, update, delete on table public.email_interests to service_role;
grant select, insert, update, delete on table public.feedback_submissions to service_role;
