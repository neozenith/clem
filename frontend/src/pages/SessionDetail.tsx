import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api';
import { Session, Event } from '../types';

export function SessionDetail() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();

  const [session, setSession] = useState<Session | null>(null);
  const [events, setEvents] = useState<Event[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    const loadSession = async () => {
      try {
        setLoading(true);
        const [sessionData, eventsData] = await Promise.all([
          api.getSession(sessionId),
          api.getSessionEvents(sessionId, { limit: 500 }),
        ]);
        setSession(sessionData);
        setEvents(eventsData);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load session');
      } finally {
        setLoading(false);
      }
    };

    loadSession();
  }, [sessionId]);

  if (loading) {
    return <div className="loading">Loading session...</div>;
  }

  if (error) {
    return (
      <div className="app">
        <div className="error">Error: {error}</div>
        <button onClick={() => navigate('/')}>Back to Dashboard</button>
      </div>
    );
  }

  if (!session) {
    return <div className="error">Session not found</div>;
  }

  return (
    <div className="app">
      <header className="app-header">
        <button
          onClick={() => navigate('/')}
          style={{
            background: 'transparent',
            border: '1px solid #444',
            color: '#888',
            padding: '0.5rem 1rem',
            borderRadius: '4px',
            cursor: 'pointer',
            marginBottom: '1rem',
          }}
        >
          ← Back to Dashboard
        </button>
        <h1 className="app-title">Session: {sessionId?.substring(0, 12)}</h1>
        <p className="app-subtitle">
          {session.project_name} • {session.domain_path}
        </p>
        <p className="app-subtitle">
          {events.length} events • Started:{' '}
          {session.started_at ? new Date(session.started_at).toLocaleString() : 'Unknown'}
        </p>
      </header>

      <div className="events-list">
        {events.length === 0 ? (
          <div className="empty-state">No events found for this session</div>
        ) : (
          events.map((event, index) => (
            <div key={index} className="event-item">
              <div className="event-header">
                <span className="event-type">{event.type || 'unknown'}</span>
                {event.role && <span className="event-role">{event.role}</span>}
                {event.timestamp && (
                  <span className="event-timestamp">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </span>
                )}
              </div>

              {event.content && (
                <div className="event-content">
                  <pre>{event.content}</pre>
                </div>
              )}

              {event.tool_name && (
                <div className="event-tool">
                  <strong>Tool:</strong> {event.tool_name}
                </div>
              )}

              {event.tool_input && (
                <div className="event-tool-input">
                  <strong>Input:</strong>
                  <pre>{event.tool_input}</pre>
                </div>
              )}

              {event.tool_output && (
                <div className="event-tool-output">
                  <strong>Output:</strong>
                  <pre>{event.tool_output}</pre>
                </div>
              )}

              {event.error && (
                <div className="event-error">
                  <strong>Error:</strong> {event.error}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
