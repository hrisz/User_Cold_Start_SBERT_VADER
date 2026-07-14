create extension if not exists pgcrypto;

create table if not exists recommendation_sessions (
    id uuid primary key default gen_random_uuid(),
    user_name text,
    keyword text not null,
    category text,
    top_n int default 10,
    model_name text,
    alpha_sbert numeric,
    beta_vader numeric,
    gamma_rating numeric,
    created_at timestamptz default now()
);

create table if not exists recommendation_items (
    id uuid primary key default gen_random_uuid(),
    session_id uuid references recommendation_sessions(id) on delete cascade,
    rank int,
    product_id text,
    product_title text,
    product_category text,
    avg_rating numeric,
    review_count int,
    sbert_similarity numeric,
    final_score numeric,
    created_at timestamptz default now()
);

alter table recommendation_sessions enable row level security;
alter table recommendation_items enable row level security;

drop policy if exists "allow insert sessions" on recommendation_sessions;
drop policy if exists "allow select sessions" on recommendation_sessions;
drop policy if exists "allow insert items" on recommendation_items;
drop policy if exists "allow select items" on recommendation_items;

create policy "allow insert sessions"
on recommendation_sessions
for insert
to anon
with check (true);

create policy "allow select sessions"
on recommendation_sessions
for select
to anon
using (true);

create policy "allow insert items"
on recommendation_items
for insert
to anon
with check (true);

create policy "allow select items"
on recommendation_items
for select
to anon
using (true);