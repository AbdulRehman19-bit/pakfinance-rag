-- PakFinance RAG — Supabase schema
-- Run once in the Supabase SQL editor (Project -> SQL Editor -> New query).

create table if not exists documents (
    document_id     text primary key,
    source          text not null,            -- 'PSX' | 'SBP' | 'FBR'
    title           text not null,
    category        text not null default 'general',
    source_url      text,
    page_count      integer,
    created_at      timestamptz not null default now()
);

create table if not exists chunks (
    chunk_id        text primary key,
    parent_id       text not null,
    document_id     text not null references documents(document_id) on delete cascade,
    source          text not null,
    category        text not null default 'general',
    document_title  text not null,
    page_number     integer,
    text            text not null,            -- child text — what gets embedded/indexed
    parent_text     text not null,            -- full parent text — what the LLM reads
    metadata        jsonb not null default '{}'::jsonb,
    created_at      timestamptz not null default now()
);

-- Makes category-filtered retrieval fast — the DB-level half of the
-- "search fewer chunks" routing strategy.
create index if not exists idx_chunks_category on chunks (category);
create index if not exists idx_chunks_source on chunks (source);
create index if not exists idx_chunks_document_id on chunks (document_id);