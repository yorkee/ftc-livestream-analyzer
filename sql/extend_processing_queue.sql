-- Extend the existing queue without replacing existing jobs.
alter table public.ftc_video_processing_jobs
  add column youtube_video_id text generated always as (
    substring(video_url from '^https?://(?:www[.]|m[.])?youtube[.]com/(?:live/|shorts/|watch[?]v=)([A-Za-z0-9_-]{11})(?:[?&#/]|$)')
    ) stored,
  add column priority integer not null default 50 check (priority between 0 and 100),
  add column duration_seconds integer check (duration_seconds > 0),
  add column video_title text,
  add column progress_percent integer not null default 0 check (progress_percent between 0 and 100),
  add column processor_version text,
  add column lease_expires_at timestamptz,
  add column completed_at timestamptz,
  add column result_summary jsonb;
create unique index ftc_jobs_unique_video on public.ftc_video_processing_jobs
  (season_year,event_code,youtube_video_id) where youtube_video_id is not null;
create index ftc_jobs_pending_priority on public.ftc_video_processing_jobs
  (priority desc,created_at,id) where status='pending' and youtube_video_id is not null;
alter table public.ftc_video_processing_jobs enable row level security;
revoke all on public.ftc_video_processing_jobs from anon, authenticated;
grant select,insert,update,delete on public.ftc_video_processing_jobs to service_role;
comment on column public.ftc_video_processing_jobs.youtube_video_id is 'Null for unresolved channel URLs or unsupported URL forms; only direct recordings are eligible for processing.';
