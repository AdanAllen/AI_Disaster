create extension if not exists pgcrypto;

create table if not exists public.cities (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  county text,
  state text default 'California',
  notes text,
  created_at timestamptz not null default now(),
  constraint cities_name_county_unique unique (name, county)
);

create table if not exists public.sources (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  agency text,
  url text,
  city text,
  county text,
  hazard_type text,
  publication_year integer,
  trust_level text,
  created_at timestamptz not null default now(),
  constraint sources_title_url_unique unique (title, url)
);

create table if not exists public.hazards (
  id uuid primary key default gen_random_uuid(),
  city_id uuid references public.cities(id) on delete set null,
  hazard_type text not null,
  risk_level text,
  explanation text,
  source_id uuid references public.sources(id) on delete set null,
  created_at timestamptz not null default now(),
  constraint hazards_city_type_source_unique unique (city_id, hazard_type, source_id)
);

create table if not exists public.document_chunks (
  id uuid primary key default gen_random_uuid(),
  source_id uuid references public.sources(id) on delete cascade,
  chunk_text text not null,
  page_number integer,
  city text,
  hazard_type text,
  citation_label text,
  review_status text default 'draft',
  created_at timestamptz not null default now()
);

create index if not exists idx_sources_city on public.sources(city);
create index if not exists idx_sources_hazard_type on public.sources(hazard_type);
create index if not exists idx_hazards_city_id on public.hazards(city_id);
create index if not exists idx_hazards_hazard_type on public.hazards(hazard_type);
create index if not exists idx_hazards_source_id on public.hazards(source_id);
create index if not exists idx_document_chunks_source_id on public.document_chunks(source_id);
create index if not exists idx_document_chunks_city on public.document_chunks(city);
create index if not exists idx_document_chunks_hazard_type on public.document_chunks(hazard_type);
create index if not exists idx_document_chunks_review_status on public.document_chunks(review_status);

alter table public.cities enable row level security;
alter table public.sources enable row level security;
alter table public.hazards enable row level security;
alter table public.document_chunks enable row level security;

drop policy if exists "Allow anon read cities" on public.cities;
drop policy if exists "Allow anon read sources" on public.sources;
drop policy if exists "Allow anon read hazards" on public.hazards;
drop policy if exists "Allow anon read document_chunks" on public.document_chunks;

create policy "Allow anon read cities"
on public.cities for select
to anon
using (true);

create policy "Allow anon read sources"
on public.sources for select
to anon
using (true);

create policy "Allow anon read hazards"
on public.hazards for select
to anon
using (true);

create policy "Allow anon read document_chunks"
on public.document_chunks for select
to anon
using (true);

insert into public.cities (name, county, state, notes)
values
  ('Alameda County', 'Alameda County', 'California', 'Countywide fallback context.'),
  ('Oakland', 'Alameda County', 'California', 'City-specific LHMP context.'),
  ('Berkeley', 'Alameda County', 'California', 'City-specific LHMP context.')
on conflict (name, county) do nothing;

insert into public.sources (title, agency, url, city, county, hazard_type, publication_year, trust_level)
values
  ('National Flood Hazard Layer', 'Federal Emergency Management Agency', 'https://www.fema.gov/flood-maps/national-flood-hazard-layer', null, 'Alameda County', 'flood', null, 'official'),
  ('Fire Hazard Severity Zones', 'CAL FIRE', 'https://osfm.fire.ca.gov/what-we-do/community-wildfire-risk-reduction/fire-hazard-severity-zones', null, 'Alameda County', 'wildfire', null, 'official'),
  ('Fault and earthquake information', 'United States Geological Survey', 'https://earthquake.usgs.gov/', null, 'Alameda County', 'earthquake', null, 'official'),
  ('City of Oakland 2021-2026 Local Hazard Mitigation Plan', 'City of Oakland', 'https://www.oaklandca.gov/Public-Safety-Streets/Fire/EMSD/Local-Hazard-Mitigation-Plan-LHMP', 'Oakland', 'Alameda County', 'all', 2021, 'official'),
  ('City of Berkeley 2024 Local Hazard Mitigation Plan', 'City of Berkeley', 'https://berkeleyca.gov/Mitigation/', 'Berkeley', 'Alameda County', 'all', 2024, 'official')
on conflict (title, url) do nothing;

insert into public.hazards (city_id, hazard_type, risk_level, explanation, source_id)
select c.id, 'earthquake', 'high', 'Local plan and regional fault context identify earthquake as a major preparedness concern.', s.id
from public.cities c, public.sources s
where c.name = 'Berkeley' and s.title = 'City of Berkeley 2024 Local Hazard Mitigation Plan'
on conflict (city_id, hazard_type, source_id) do nothing;

insert into public.hazards (city_id, hazard_type, risk_level, explanation, source_id)
select c.id, 'wildfire', 'high', 'Local plan context identifies wildland-urban-interface fire and evacuation constraints as important Berkeley concerns.', s.id
from public.cities c, public.sources s
where c.name = 'Berkeley' and s.title = 'City of Berkeley 2024 Local Hazard Mitigation Plan'
on conflict (city_id, hazard_type, source_id) do nothing;

insert into public.hazards (city_id, hazard_type, risk_level, explanation, source_id)
select c.id, 'earthquake', 'high', 'Oakland local hazard planning identifies earthquake as a major citywide hazard.', s.id
from public.cities c, public.sources s
where c.name = 'Oakland' and s.title = 'City of Oakland 2021-2026 Local Hazard Mitigation Plan'
on conflict (city_id, hazard_type, source_id) do nothing;
