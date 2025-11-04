# CLEM Project Roadmap

Track progress toward the vision outlined in [NEW_VISION.md](NEW_VISION.md).

## Current Status: Phase 2 Complete ✅

**Overall Progress**: ~30% complete toward full vision

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
  - Centralized paths: ~/.clem/memory.duckdb (NEW location)
  - Schema versioning (v1.0.0)
  - Embedding configuration for future memory extraction

- [x] **Project discovery** (src/clem/core/project.py)
  - DuckDB-based JSONL parsing of session files
  - CWD extraction from Claude Code sessions
  - Session metadata aggregation (event counts, timestamps, git branch)

### Architecture Decisions
- **Disposable database**: Can be deleted and rebuilt from source at any time
- **Breaking changes**: New database location (~/.clem/memory.duckdb)
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
  - **100% test coverage** on all query modules

- [x] **Display layer** (src/clem/display/)
  - Rich table formatters: domains, projects, sessions, stats
  - Centralized presentation logic
  - **98% test coverage**

- [x] **CLI refactoring** (src/clem/cli.py)
  - Separated concerns: CLI → Queries → Display
  - Reduced from 165 lines to 120 lines (-27%)
  - **67% coverage** via integration tests

- [x] **Test suite**
  - 86 tests passing (39 → 86, +121% increase)
  - **80% overall coverage** (exceeded >75% target)
  - test_config, test_domain, test_database, test_queries, test_display, test_cli_integration

### Architecture Benefits
- **Testable**: Each layer independently testable
- **Maintainable**: Clear module responsibilities
- **Reusable**: Queries + Display ready for Web UI integration
- **Type-safe**: Full mypy compliance

---

## 🚧 Phase 3: Web UI & Observability (IN PROGRESS - Backend Complete ✅)

**Goal**: Visual interface for exploring sessions and events.

**From NEW_VISION.md**:
> `uvx clem web` should start a local web server which will start a locally run FastAPI webserver which will host our frontend webapp. It will be a plain React application built with Vite that interacts the FastAPI webserver which holds the logic for interacting with the knowledge base we will be curating.

### Completed Work ✅
- [x] **FastAPI backend** (src/clem/web/)
  - REST API endpoints: /api/stats, /api/domains, /api/projects, /api/sessions
  - Pydantic models for request/response validation
  - CORS configuration for local React dev server (localhost:5173, localhost:3000)
  - OpenAPI/Swagger documentation at /api/docs
  - Health check endpoint at /api/health

- [x] **CLI integration**
  - `clem web` command to start uvicorn server
  - Port configuration (--port, default: 8000)
  - Host configuration (--host, default: 127.0.0.1)
  - Auto-open browser option (--no-browser to disable)

- [x] **API Tests**
  - 16 new endpoint tests (102 total tests passing)
  - 81% overall coverage (up from 80%)
  - Full test coverage for all API routes

### Remaining Work
- [ ] **React frontend** (frontend/)
  - Vite + React setup
  - Domain/Project/Session hierarchy navigation
  - Session event timeline visualization
  - Live event streaming (optional WebSocket support)

### Architecture Decisions
- ✅ Leveraged existing queries/ layer for business logic (no duplication)
- ✅ FastAPI purely for API routing (clean separation)
- ✅ REST-first approach (WebSocket deferred to future)
- ✅ Type-safe Pydantic models
- ✅ Monkeypatch-based testing with test database isolation

---

## 🔮 Phase 4: MCP Server Integration (PLANNED)

**Goal**: Enable Claude Code to retrieve knowledge at Project/Domain/Global levels.

**From NEW_VISION.md**:
> I want CLEM to be able to run as an MCP for Claude Code. Claude Code should be able retrieve knowledge at Project Level, Domain Level and Global Level

### Planned Work
- [ ] **MCP server implementation** (src/clem/mcp/)
  - MCP protocol compliance
  - Resource providers: project, domain, global context
  - Tool definitions for memory search

- [ ] **Query scoping**
  - Project-level memory retrieval
  - Domain-level pattern aggregation
  - Global cross-domain insights

- [ ] **CLI integration**
  - `clem mcp` command to start MCP server
  - Configuration for Claude Code integration

### Dependencies
- Phase 5 (memory extraction) for rich context retrieval
- Existing queries layer provides foundation

---

## 🧠 Phase 5: Memory Extraction & Knowledge Graphs (PLANNED)

**Goal**: Extract memories, requirements, and build knowledge graphs.

**From NEW_VISION.md**:
> We should start with Agentic Memory extraction and consolidation using huggingface models. We should also form base level knowledge graphs. We should also create hierarchical layers of community detection on the knowledge graph. We should use GraphRAG which also needs Node2Vec to be able to search a graph across a vector space.

### Planned Work - Memory Extraction
- [ ] **Agentic memory extraction** (src/clem/memory/)
  - HuggingFace model integration (sentence-transformers configured)
  - Session event processing pipeline
  - Memory consolidation and deduplication
  - Store in memories table with VSS embeddings

- [ ] **Requirements extraction**
  - Identify implicit requirements from user messages
  - Extract "should do X" patterns
  - Auto-generate CLAUDE.md updates (future)

### Planned Work - Knowledge Graphs
- [ ] **Graph construction** (src/clem/graph/)
  - Entity extraction from memories
  - Relationship detection
  - Store graph in DuckDB (nodes + edges tables)

- [ ] **GraphRAG pipeline**
  - Node2Vec embeddings for graph search
  - Community detection (hierarchical clustering)
  - Vector space graph traversal

- [ ] **Query integration**
  - Semantic search over memories
  - Graph-based retrieval augmentation
  - Multi-hop reasoning support

### Database Schema Updates
- Memories table already exists with VSS column
- Add: graph_nodes, graph_edges, communities tables
- Add: memory_sources, memory_relationships tables

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
- Phases 4 & 5 (MCP + memory extraction) required
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
- **Tests**: 86 passing
- **Coverage**: 80% overall
  - Core modules: 97-100%
  - Database: 73-100%
  - Queries: 100%
  - Display: 98%
  - CLI: 67%
- **Linting**: 0 warnings (ruff + mypy)
- **Type safety**: 100% (mypy strict)

### Codebase Size
- **Modules**: 16 source files
- **Lines of Code**: ~1,500 (down from 1,774 monolith)
- **Test Files**: 6 files
- **Test Lines**: ~800

### Database
- **Schema Version**: 1.0.0
- **Tables**: 5 (domains, projects, sessions, memories, metadata)
- **Indexes**: Primary keys + VSS HNSW on memories

---

## Next Steps

### Immediate (Phase 3)
1. Setup FastAPI backend structure
2. Create React + Vite frontend skeleton
3. Implement basic domain/project/session browsing
4. Add `clem web` command

### Short-term (Phase 4)
1. Research MCP protocol requirements
2. Design resource providers
3. Implement basic MCP server
4. Test Claude Code integration

### Medium-term (Phase 5)
1. Implement memory extraction pipeline
2. Build knowledge graph construction
3. Add GraphRAG capabilities
4. Semantic search over memories

### Long-term (Phase 6)
1. Lessons extraction from sessions
2. .claude/ incremental updates
3. Learning feedback loops

---

## Questions & Decisions Needed

### Open Questions
1. **Web UI scope**: How sophisticated should the initial UI be? Start with read-only views?
2. **MCP priority**: Should MCP come before or after Web UI?
3. **Memory models**: Which HuggingFace models for memory extraction? (sentence-transformers/all-MiniLM-L6-v2 configured)
4. **Graph DB**: DuckDB vs. dedicated graph database (Neo4j)?
5. **Learning triggers**: When should incremental learning happen? Manual trigger vs. automatic?

### Design Decisions
- **Database location**: ~/.clem/memory.duckdb (confirmed)
- **Domain hierarchy**: Path-based detection (implemented)
- **Database pattern**: Disposable cache (confirmed)
- **Test coverage target**: >75% pragmatic coverage (achieved: 80%)

---

## Timeline Estimate

- **Phase 1 & 2**: ✅ Complete (2 sessions)
- **Phase 3** (Web UI): ~2-3 sessions
- **Phase 4** (MCP): ~2 sessions
- **Phase 5** (Memory/Graph): ~4-5 sessions
- **Phase 6** (Learning): ~3-4 sessions
- **Phase 7** (Distribution): ~1 session

**Total estimated**: ~14-17 sessions remaining to full vision (~85% complete)

---

*Last updated: 2025-11-05 - Phase 2 complete, starting Phase 3 planning*
