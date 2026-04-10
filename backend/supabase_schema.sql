create table if not exists public.drug_reviews (
    id bigserial primary key,
    drugName text not null,
    condition text not null,
    review text not null,
    usefulCount integer not null default 0,
    rating numeric,
    response_category integer,
    created_at timestamptz not null default now()
);

create index if not exists idx_drug_reviews_drug_name
    on public.drug_reviews (drugName);

create index if not exists idx_drug_reviews_condition
    on public.drug_reviews (condition);

create index if not exists idx_drug_reviews_response_category
    on public.drug_reviews (response_category);
