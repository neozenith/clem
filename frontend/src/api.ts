/**
 * API client for CLEM backend
 */

import { Stats, Domain, Project, Session, Event } from './types';

const API_BASE = '/api';

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  // Stats
  getStats: () => fetchJson<Stats>(`${API_BASE}/stats`),

  // Domains
  listDomains: () => fetchJson<Domain[]>(`${API_BASE}/domains`),
  getDomain: (domainId: string) => fetchJson<Domain>(`${API_BASE}/domains/${domainId}`),

  // Projects
  listProjects: (domainId?: string) => {
    const params = domainId ? `?domain_id=${encodeURIComponent(domainId)}` : '';
    return fetchJson<Project[]>(`${API_BASE}/projects${params}`);
  },
  getProject: (projectId: string) => fetchJson<Project>(`${API_BASE}/projects/${projectId}`),

  // Sessions
  listSessions: (options?: { projectName?: string; domainId?: string; limit?: number }) => {
    const params = new URLSearchParams();
    if (options?.projectName) params.append('project_name', options.projectName);
    if (options?.domainId) params.append('domain_id', options.domainId);
    if (options?.limit) params.append('limit', options.limit.toString());

    const query = params.toString();
    return fetchJson<Session[]>(`${API_BASE}/sessions${query ? `?${query}` : ''}`);
  },
  getSession: (sessionId: string) => fetchJson<Session>(`${API_BASE}/sessions/${sessionId}`),

  // Events
  getSessionEvents: (
    sessionId: string,
    options?: {
      limit?: number;
      offset?: number;
    }
  ) => {
    const params = new URLSearchParams();
    if (options?.limit) params.append('limit', options.limit.toString());
    if (options?.offset) params.append('offset', options.offset.toString());

    const query = params.toString();
    return fetchJson<Event[]>(
      `${API_BASE}/sessions/${sessionId}/events${query ? `?${query}` : ''}`
    );
  },
};
