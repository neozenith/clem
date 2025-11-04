/**
 * Type definitions matching the FastAPI backend models
 */

export interface Stats {
  domains: number;
  projects: number;
  sessions: number;
  memories: number;
  schema_version: string;
  last_rebuild: string | null;
}

export interface Domain {
  domain_id: string;
  domain_path: string;
  project_count: number;
  session_count: number;
}

export interface Project {
  project_name: string;
  domain_id: string;
  session_count: number;
  cwd: string;
}

export interface Session {
  session_id: string;
  project_name: string;
  domain_path: string;
  event_count: number;
  started_at: string | null;
}
