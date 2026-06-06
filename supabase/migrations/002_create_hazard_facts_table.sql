create table if not exists public.hazard_facts (
  id text primary key,
  jurisdiction text not null,
  county text default 'Alameda County',
  hazard text not null,
  location_cue text,
  resident_meaning text,
  before_actions text[] default '{}',
  during_actions text[] default '{}',
  after_actions text[] default '{}',
  recovery_steps text[] default '{}',
  household_considerations text[] default '{}',
  source_name text,
  source_url text,
  evidence_type text,
  review_status text default 'draft',
  specificity_level text default 'county',
  last_updated text,
  created_at timestamptz not null default now()
);

create index if not exists idx_hazard_facts_jurisdiction on public.hazard_facts(jurisdiction);
create index if not exists idx_hazard_facts_hazard on public.hazard_facts(hazard);
create index if not exists idx_hazard_facts_review_status on public.hazard_facts(review_status);

alter table public.hazard_facts enable row level security;

drop policy if exists "Allow anon read hazard_facts" on public.hazard_facts;

create policy "Allow anon read hazard_facts"
on public.hazard_facts for select
to anon
using (true);
