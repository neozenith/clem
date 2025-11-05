# CLEM Project Roadmap

Track progress toward the vision outlined in [NEW_VISION.md](NEW_VISION.md).

## Current Status: Phase 3 Complete ✅

**Overall Progress**: ~40% complete toward full vision

---

## ✅ Phase 1: Modular Architecture (COMPLETE)

**Goal**: Refactor monolithic script into maintainable, testable modules.

### Completed Work
- [x] **Core domain logic** (src/clem/core/)
  - Domain extraction algorithm: path-based hierarchy detection
  - Project ID generation with domain awareness
  - Claude Code path encoding compatibility

- [x] **Database layer** (src/clem/database/)
  - DuckDB connection management with lazy initialization
  - Domain-aware schema: domains → projects → sessions → events → memories
  - VSS extension setup with HNSW experimental persistence
  - Nuclear rebuild capability (disposable cache pattern)

- [x] **Configuration management** (src/clem/config.py)
  - Centralized paths: ~/.clem/memory.duckdb
  - Schema versioning (v1.0.0)
  - Embedding configuration for future memory extraction

- [x] **Project discovery** (src/clem/core/project.py)
  - DuckDB-based JSONL parsing of session files
  - CWD extraction from Claude Code sessions
  - Session metadata aggregation (event counts, timestamps, git branch)

### Architecture Decisions
- **Disposable database**: Can be deleted and rebuilt from source at any time
- **Database location**: ~/.clem/memory.duckdb
- **DuckDB native**: Direct JSONL reading, no external ETL required
- **Domain hierarchy**: play/, work/, clients/acmeinc/, foss/ segregation

---

## ✅ Phase 2: Quality Infrastructure & Layered Architecture (COMPLETE)

**Goal**: Achieve >75% test coverage with clean separation of concerns.

### Completed Work
- [x] **Linting infrastructure**
  - ruff (linting + formatting)
  - mypy (type checking)
  - make lint target (0 warnings)

- [x] **Queries layer** (src/clem/queries/)
  - DomainQuery: list_all, get_by_id, count
  - ProjectQuery: filtering by domain
  - SessionQuery: filtering by project/domain with pagination
  - EventQuery: read events from session JSONL files
  - **100% test coverage** on all query modules

- [x] **Display layer** (src/clem/display/)
  - Rich table formatters: domains, projects, sessions, stats
  - Centralized presentation logic
  - **98% test coverage**

- [x] **CLI refactoring** (src/clem/cli.py)
  - Separated concerns: CLI → Queries → Display
  - **67% coverage** via integration tests

- [x] **Test suite**
  - 86+ tests passing
  - **80%+ overall coverage**
  - test_config, test_domain, test_database, test_queries, test_display, test_cli_integration

### Architecture Benefits
- **Testable**: Each layer independently testable
- **Maintainable**: Clear module responsibilities
- **Reusable**: Queries + Display ready for Web UI integration
- **Type-safe**: Full mypy compliance

---

## ✅ Phase 3: Web UI & Observability (COMPLETE)

**Goal**: Visual interface for exploring sessions and events.

**From NEW_VISION.md**:
> `uvx clem web` should start a local web server which will start a locally run FastAPI webserver which will host our frontend webapp. It will be a plain React application built with Vite that interacts the FastAPI webserver which holds the logic for interacting with the knowledge base we will be curating.

### Completed Work
- [x] **FastAPI backend** (src/clem/web/)
  - REST API endpoints: /api/stats, /api/domains, /api/projects, /api/sessions, /api/sessions/{id}/events
  - Pydantic models for request/response validation
  - CORS configuration for local React dev server
  - OpenAPI/Swagger documentation at /api/docs
  - Health check endpoint at /api/health
  - Static file serving for React build
  - Client-side routing support

- [x] **React frontend** (frontend/)
  - Vite + React 18 + TypeScript setup
  - React Router for URL-based navigation
  - Domain/Project/Session hierarchy navigation
  - Session event timeline visualization (turn-by-turn)
  - Tailwind CSS for styling
  - Storybook for component development
  - Vitest + Playwright for testing

- [x] **CLI integration**
  - `clem web` command to start uvicorn server
  - Port configuration (--port, default: 8000)
  - Host configuration (--host, default: 127.0.0.1)
  - Auto-open browser option (--no-browser to disable)

- [x] **EventQuery with flexible schema handling**
  - JSON extraction from nested session file structure
  - Handles varying JSONL schemas gracefully
  - Filters for user/assistant message types

- [x] **API Tests & Frontend Tests**
  - 102+ total tests passing
  - 81% overall coverage
  - Full test coverage for all API routes
  - Frontend component tests with Vitest

### Architecture Decisions
- ✅ Leveraged existing queries/ layer for business logic (no duplication)
- ✅ FastAPI purely for API routing (clean separation)
- ✅ REST-first approach
- ✅ Type-safe Pydantic models
- ✅ Client-side routing with React Router
- ✅ Monkeypatch-based testing with test database isolation

---

## 🚧 Phase 4: Memory Extraction & Knowledge Graphs (IN PROGRESS)

**Goal**: Extract memories, build knowledge graphs, and visualize them in the web UI.

**Strategic Decision**: Build memory extraction and knowledge graph **before** MCP server to validate data flows and knowledge representation. This ensures we have a solid foundation of knowledge to expose via MCP later.

**From NEW_VISION.md**:
> We should start with Agentic Memory extraction and consolidation using huggingface models. We should also form base level knowledge graphs. We should also create hierarchical layers of community detection on the knowledge graph. We should use GraphRAG which also needs Node2Vec to be able to search a graph across a vector space.

### Phase 4A: Memory Extraction
- [ ] **Memory extraction pipeline** (src/clem/memory/)
  - HuggingFace model integration (sentence-transformers for embeddings)
  - Session event processing pipeline
  - Entity and concept extraction from conversations
  - Memory consolidation and deduplication using vector similarity
  - Store in memories table with VSS embeddings (384-dimensional)

- [ ] **Memory types**
  - Preferences (coding style, tool preferences, workflow patterns)
  - Decisions (architectural choices, design patterns)
  - Requirements (implicit "should do X" patterns)
  - Knowledge (learned concepts, techniques, solutions)

- [ ] **Memory API endpoints** (src/clem/web/routers/memories.py)
  - GET /api/memories - list memories with filtering
  - GET /api/memories/{id} - get specific memory
  - POST /api/memories/extract - trigger extraction for session/project
  - GET /api/memories/search - semantic search

### Phase 4B: Knowledge Graph Construction
- [ ] **Graph construction** (src/clem/graph/)
  - Entity extraction from memories and conversations
  - Relationship detection between entities
  - Store graph in DuckDB (graph_nodes + graph_edges tables)
  - Node types: concepts, techniques, patterns, problems, solutions
  - Edge types: relates_to, solves, uses, implements, requires

- [ ] **Graph algorithms**
  - Community detection (hierarchical clustering of related concepts)
  - Centrality metrics (identify important concepts)
  - Path finding (discover connections between concepts)
  - Subgraph extraction (focus on relevant portions)

- [ ] **GraphRAG pipeline**
  - Node2Vec embeddings for graph search
  - Vector space graph traversal
  - Multi-hop reasoning support
  - Retrieval-augmented generation over graph

### Phase 4C: Web Visualization
- [ ] **Graph visualization component** (frontend/src/components/GraphView/)
  - Interactive graph rendering (D3.js or React Flow)
  - Node filtering by type, domain, project
  - Zoom, pan, and focus navigation
  - Edge highlighting and relationship exploration
  - Community visualization (colored clusters)

- [ ] **Memory explorer** (frontend/src/pages/MemoryExplorer.tsx)
  - Memory list view with search and filtering
  - Memory detail view with linked entities
  - Timeline view of memory creation
  - Graph view showing memory relationships

- [ ] **Graph API endpoints** (src/clem/web/routers/graph.py)
  - GET /api/graph/nodes - list nodes with filtering
  - GET /api/graph/edges - list edges
  - GET /api/graph/subgraph - get subgraph around entity
  - GET /api/graph/communities - get community clusters
  - GET /api/graph/search - semantic search over graph

### Database Schema Updates
- [ ] **Memories table enhancements**
  - memory_id, content, embedding (VSS), confidence
  - session_id, project_id, domain_id (lineage)
  - entity_type, entity_id (linked to graph)
  - created_at, updated_at, version

- [ ] **Graph tables**
  - graph_nodes: node_id, node_type, label, properties, embedding
  - graph_edges: edge_id, source_id, target_id, edge_type, weight
  - graph_communities: community_id, node_id, level (hierarchical)
  - node_embeddings: node_id, embedding (Node2Vec)

- [ ] **Indexes**
  - VSS HNSW on memories.embedding
  - VSS HNSW on graph_nodes.embedding
  - VSS HNSW on node_embeddings.embedding
  - B-tree on graph_edges (source_id, target_id)

### Dependencies & Models
- **sentence-transformers/all-MiniLM-L6-v2**: Memory embeddings (384-dim)
- **spaCy or similar**: Entity extraction
- **Node2Vec**: Graph embeddings for traversal
- **scikit-learn**: Community detection algorithms
- **DuckDB VSS extension**: Vector similarity search

### Testing Strategy
- Unit tests for extraction pipeline
- Integration tests for graph construction
- API tests for memory/graph endpoints
- Frontend tests for visualization components
- End-to-end tests for full extraction → visualization flow

---

## 🔮 Phase 5: MCP Server Integration (PLANNED - Deferred)

**Goal**: Enable Claude Code to retrieve knowledge at Project/Domain/Global levels.

**Strategic Decision**: Build MCP **after** memory extraction and knowledge graph to ensure we have validated data flows and rich knowledge to expose.

**From NEW_VISION.md**:
> I want CLEM to be able to run as an MCP for Claude Code. Claude Code should be able retrieve knowledge at Project Level, Domain Level and Global Level

### Planned Work
- [ ] **MCP server implementation** (src/clem/mcp/)
  - MCP protocol compliance
  - Resource providers: project, domain, global context
  - Tool definitions for memory search
  - Graph-based context retrieval

- [ ] **Query scoping**
  - Project-level memory retrieval
  - Domain-level pattern aggregation
  - Global cross-domain insights
  - Graph traversal for related concepts

- [ ] **CLI integration**
  - `clem mcp` command to start MCP server
  - Configuration for Claude Code integration
  - Multi-session context loading

### Dependencies
- **Phase 4 (memory + graph) MUST be complete first**
- Existing queries layer provides foundation
- Graph structure enables rich context retrieval

---

## 🎓 Phase 6: Incremental Learning (FUTURE)

**Goal**: Enable Claude Code to improve itself via .claude/ updates.

**From NEW_VISION.md**:
> This incremental learning mechanism will need to check the live documentation of Claude Code. I imagine this would work by creating a slash command for claude code that would leverage the MCP for latest insights and then prompt claude code with tactical changes that get made to that project's .claude folder.

### Planned Work
- [ ] **Lessons extraction** (src/clem/lessons/)
  - Analyze successful sessions for patterns
  - Extract reusable .claude/lessons/ markdown
  - Scope lessons: project-specific vs. domain-wide vs. global

- [ ] **Claude Code integration**
  - Slash command: `/clem-learn` or similar
  - Prompt templates for .claude/ updates
  - Version control awareness for shared .claude repos

- [ ] **Learning feedback loop**
  - Track lesson effectiveness
  - Refine based on usage patterns
  - Global abstraction promotion

### Dependencies
- Phases 4 & 5 (memory/graph + MCP) required
- Live Claude Code documentation integration needed

---

## 📦 Phase 7: Distribution & Installation (PARTIAL)

**Goal**: Easy installation and updates.

**From NEW_VISION.md**:
> `clem` should be installable via `uv tool install git+https://github.com/neozenith/clem@main` and then the cli can be run via `uvx clem`

### Current Status
- [x] pyproject.toml configured with CLI entry point
- [x] Can run via `uv run clem` in development
- [ ] Publish to GitHub with proper tags
- [ ] Test `uv tool install` workflow
- [ ] Setup CI/CD for releases
- [ ] Documentation for installation

---

## Metrics Dashboard

### Code Quality
- **Tests**: 102+ passing
- **Coverage**: 81% overall
  - Core modules: 97-100%
  - Database: 73-100%
  - Queries: 100%
  - Display: 98%
  - CLI: 67%
  - Web: 85%
- **Linting**: 0 warnings (ruff + mypy)
- **Type safety**: 100% (mypy strict)

### Codebase Size
- **Modules**: 20+ source files
- **Lines of Code**: ~2,500 (backend + frontend)
- **Test Files**: 8+ files
- **Test Lines**: ~1,000+

### Database
- **Schema Version**: 1.0.0
- **Tables**: 5 (domains, projects, sessions, memories, metadata)
- **Indexes**: Primary keys + VSS HNSW on memories.embedding
- **Future**: +4 tables for knowledge graph (nodes, edges, communities, embeddings)

---

## Next Steps

### Immediate (Phase 4A - Memory Extraction)
1. Design memory extraction pipeline architecture
2. Integrate sentence-transformers for embeddings
3. Implement entity extraction from conversations
4. Build memory consolidation with vector similarity
5. Create memory API endpoints
6. Add memory management UI

### Short-term (Phase 4B - Knowledge Graph)
1. Design graph schema (nodes, edges, communities)
2. Implement entity extraction and relationship detection
3. Build graph construction pipeline
4. Integrate Node2Vec for graph embeddings
5. Implement community detection algorithms
6. Create graph API endpoints

### Medium-term (Phase 4C - Graph Visualization)
1. Choose graph visualization library (D3.js vs React Flow)
2. Build interactive graph component
3. Implement memory explorer UI
4. Add graph filtering and search
5. Create community visualization
6. End-to-end testing of extraction → visualization

### Long-term (Phase 5 & Beyond)
1. MCP server implementation
2. Claude Code integration
3. Incremental learning system
4. Distribution and CI/CD

---

## Questions & Decisions

### Open Questions
1. **Entity extraction**: Use spaCy, custom NER, or LLM-based extraction?
2. **Graph visualization**: D3.js (flexible) vs React Flow (easier) vs Cytoscape?
3. **Memory consolidation**: What similarity threshold for deduplication? (0.85-0.90?)
4. **Community detection**: Which algorithm? (Louvain, Label Propagation, or Hierarchical?)
5. **Node2Vec parameters**: Dimensions (128?), walk length, number of walks?

### Design Decisions
- **Database location**: ~/.clem/memory.duckdb ✅
- **Domain hierarchy**: Path-based detection ✅
- **Database pattern**: Disposable cache ✅
- **Test coverage target**: >75% ✅ (achieved: 81%)
- **Phase order**: Memory/Graph before MCP ✅
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2 (384-dim) ✅

---

## Timeline Estimate

- **Phase 1 & 2**: ✅ Complete (2 sessions)
- **Phase 3** (Web UI): ✅ Complete (1 session)
- **Phase 4A** (Memory Extraction): ~2-3 sessions
- **Phase 4B** (Knowledge Graph): ~3-4 sessions
- **Phase 4C** (Graph Visualization): ~2-3 sessions
- **Phase 5** (MCP): ~2 sessions
- **Phase 6** (Learning): ~3-4 sessions
- **Phase 7** (Distribution): ~1 session

**Total estimated**: ~13-17 sessions remaining to full vision (~60% complete with Phase 3 done)

---

*Last updated: 2025-11-05 - Phase 3 complete, starting Phase 4 (Memory & Knowledge Graphs)*
