-- RoboHub: tasks, physics-gated demonstrations, SmolVLA checkpoints, trajectory search.

create extension if not exists vector with schema extensions;

create table if not exists public.tasks (
  id uuid primary key default gen_random_uuid(),
  prompt text not null,
  arm_type text not null default 'so101',
  status text not null default 'queued'
    check (status in ('queued', 'filming', 'gating', 'accepted', 'training', 'done', 'failed')),
  created_at timestamptz not null default now()
);

create table if not exists public.demonstrations (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null references public.tasks (id) on delete cascade,
  video_storage_path text,
  passed_gates boolean,
  gate_audit jsonb not null default '{}'::jsonb,
  trajectory_vector extensions.vector(1536),
  dataset_path text,
  created_at timestamptz not null default now()
);

create table if not exists public.models (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null references public.tasks (id) on delete cascade,
  model_name text not null,
  weights_url text,
  eval_score double precision,
  training_cost double precision,
  created_at timestamptz not null default now()
);

create index if not exists demonstrations_task_id_idx on public.demonstrations (task_id);
create index if not exists models_task_id_idx on public.models (task_id);
create index if not exists demonstrations_trajectory_idx
  on public.demonstrations
  using hnsw (trajectory_vector extensions.vector_cosine_ops);

-- Cosine distance (<=>). Lower distance means a closer 3D trajectory embedding.
create or replace function public.search_similar_trajectories(
  query_embedding extensions.vector(1536),
  match_count integer default 8,
  task_filter uuid default null
)
returns table (
  id uuid,
  task_id uuid,
  dataset_path text,
  passed_gates boolean,
  distance double precision
)
language sql
stable
as $$
  select
    d.id,
    d.task_id,
    d.dataset_path,
    d.passed_gates,
    (d.trajectory_vector <=> query_embedding)::double precision as distance
  from public.demonstrations d
  where d.trajectory_vector is not null
    and (task_filter is null or d.task_id = task_filter)
  order by d.trajectory_vector <=> query_embedding
  limit greatest(match_count, 1);
$$;

alter table public.tasks enable row level security;
alter table public.demonstrations enable row level security;
alter table public.models enable row level security;

create policy tasks_read on public.tasks for select using (true);
create policy demonstrations_read on public.demonstrations for select using (true);
create policy models_read on public.models for select using (true);

do $$
begin
  alter publication supabase_realtime add table public.tasks;
exception
  when duplicate_object then null;
  when undefined_object then null;
end $$;

do $$
begin
  alter publication supabase_realtime add table public.demonstrations;
exception
  when duplicate_object then null;
  when undefined_object then null;
end $$;

insert into storage.buckets (id, name, public)
values ('robohub-artifacts', 'robohub-artifacts', false)
on conflict (id) do nothing;
