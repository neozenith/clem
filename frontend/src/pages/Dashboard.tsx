import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import { Stats, Domain, Project, Session } from '../types';

export function Dashboard() {
  const navigate = useNavigate();
  const [stats, setStats] = useState<Stats | null>(null);
  const [domains, setDomains] = useState<Domain[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);

  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load initial data
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const [statsData, domainsData] = await Promise.all([api.getStats(), api.listDomains()]);
        setStats(statsData);
        setDomains(domainsData);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  // Load projects when domain is selected
  useEffect(() => {
    if (!selectedDomain) {
      setProjects([]);
      return;
    }

    const loadProjects = async () => {
      try {
        const projectsData = await api.listProjects(selectedDomain);
        setProjects(projectsData);
      } catch (err) {
        console.error('Failed to load projects:', err);
      }
    };
    loadProjects();
  }, [selectedDomain]);

  // Load sessions when project is selected
  useEffect(() => {
    if (!selectedProject) {
      setSessions([]);
      return;
    }

    const loadSessions = async () => {
      try {
        const sessionsData = await api.listSessions({ projectName: selectedProject });
        setSessions(sessionsData);
      } catch (err) {
        console.error('Failed to load sessions:', err);
      }
    };
    loadSessions();
  }, [selectedProject]);

  const handleDomainClick = (domainId: string) => {
    if (selectedDomain === domainId) {
      setSelectedDomain(null);
      setSelectedProject(null);
    } else {
      setSelectedDomain(domainId);
      setSelectedProject(null);
    }
  };

  const handleProjectClick = (projectName: string) => {
    if (selectedProject === projectName) {
      setSelectedProject(null);
    } else {
      setSelectedProject(projectName);
    }
  };

  const handleSessionClick = (sessionId: string) => {
    navigate(`/sessions/${sessionId}`);
  };

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">CLEM</h1>
        <p className="app-subtitle">Claude Learning & Experience Manager</p>
      </header>

      {stats && (
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-label">Domains</div>
            <div className="stat-value">{stats.domains}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Projects</div>
            <div className="stat-value">{stats.projects}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Sessions</div>
            <div className="stat-value">{stats.sessions}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Memories</div>
            <div className="stat-value">{stats.memories}</div>
          </div>
        </div>
      )}

      <h2 className="section-title">Domains</h2>
      {domains.length === 0 ? (
        <div className="empty-state">
          No domains found. Run <code>clem rebuild</code> to populate the database.
        </div>
      ) : (
        <div className="domains-grid">
          {domains.map((domain) => (
            <div
              key={domain.domain_id}
              className={`domain-card ${selectedDomain === domain.domain_id ? 'selected' : ''}`}
              onClick={() => handleDomainClick(domain.domain_id)}
            >
              <div className="domain-header">
                <h3 className="domain-name">{domain.domain_path || '(no domain)'}</h3>
                <div className="domain-counts">
                  <span>{domain.project_count} projects</span>
                  <span>{domain.session_count} sessions</span>
                </div>
              </div>

              {selectedDomain === domain.domain_id && projects.length > 0 && (
                <div className="projects-list">
                  {projects.map((project) => (
                    <div
                      key={project.project_name}
                      className={`project-item ${selectedProject === project.project_name ? 'selected' : ''}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleProjectClick(project.project_name);
                      }}
                    >
                      <div className="project-name">{project.project_name}</div>
                      <div className="project-info">
                        {project.session_count} sessions • {project.cwd}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {selectedProject && sessions.length > 0 && (
        <>
          <h2 className="section-title">Sessions for {selectedProject}</h2>
          <div className="sessions-list">
            {sessions.map((session) => (
              <div
                key={session.session_id}
                className="session-item"
                onClick={() => handleSessionClick(session.session_id)}
                style={{ cursor: 'pointer' }}
              >
                <div className="session-header">
                  <span className="session-id">{session.session_id.substring(0, 8)}</span>
                  <span className="session-info">{session.event_count} events</span>
                </div>
                <div className="session-info">
                  {session.started_at
                    ? new Date(session.started_at).toLocaleString()
                    : 'No start time'}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
