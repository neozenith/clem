# CLEM Frontend

React + TypeScript + Vite frontend for CLEM (Claude Learning & Experience Manager).

## Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Development

The frontend connects to the FastAPI backend running on `http://localhost:8000`.

Make sure the backend is running:
```bash
# From the project root
uv run clem web
```

Then start the frontend dev server:
```bash
# From the frontend directory
npm run dev
```

Visit http://localhost:5173 to view the app.

## Features

- **Dashboard**: Overview statistics for domains, projects, sessions, and memories
- **Domain Navigation**: Browse domains with project counts and session counts
- **Project View**: Expand domains to see projects with session information
- **Session List**: View sessions for selected projects with event counts and timestamps

## Tech Stack

- **React 18**: UI framework
- **TypeScript**: Type safety
- **Vite**: Fast build tool and dev server
- **React Router**: Client-side routing (future)
- **Native CSS**: Simple styling (no framework dependencies)

## API Integration

The frontend uses a type-safe API client (`src/api.ts`) that mirrors the FastAPI backend models defined in `src/types.ts`.

All API calls are proxied through Vite's dev server to avoid CORS issues during development.
