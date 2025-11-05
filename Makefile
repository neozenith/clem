# CLEM - Sleep Cycle Makefile

.PHONY: help docs
.PHONY: frontend-install frontend-dev frontend-build frontend-lint frontend-format frontend-test
.PHONY: frontend-test-watch frontend-test-coverage frontend-storybook frontend-storybook-build

# Ensure the .make folder exists when starting make
# We need this for build targets that have multiple or no file output.
# We 'touch' files in here to mark the last time the specific job completed.
_ := $(shell mkdir -p .make)
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

init: .make/init
.make/init:
	uv sync --extra dev
	@touch $@

# Frontend targets
frontend-install: .make/frontend-install
.make/frontend-install:
	@echo "📦 Installing frontend dependencies..."
	npm --prefix frontend install
	@touch $@

frontend-dev: .make/frontend-install
	@echo "🚀 Starting frontend dev server..."
	npm --prefix frontend run dev

frontend-build: .make/frontend-install
	@echo "🏗️  Building frontend for production..."
	npm --prefix frontend run build
	@echo "✓ Frontend built to src/clem/web/frontend/"

frontend-lint: .make/frontend-install
	@echo "🔍 Linting frontend code (includes format check & typecheck)..."
	npm --prefix frontend run lint
	@echo "✓ Frontend linting complete!"

frontend-format: .make/frontend-install
	@echo "🎨 Formatting frontend code..."
	npm --prefix frontend run format
	@echo "✓ Frontend formatting complete!"

frontend-test: .make/frontend-install
	@echo "🧪 Running frontend tests (Vitest + Playwright)..."
	npm --prefix frontend run test
	@echo "✓ Frontend tests complete!"

frontend-test-watch: .make/frontend-install
	@echo "🧪 Running frontend tests in watch mode..."
	npm --prefix frontend run test:watch

frontend-test-coverage: .make/frontend-install
	@echo "📊 Running frontend tests with coverage..."
	npm --prefix frontend run test:coverage

frontend-storybook: .make/frontend-install
	@echo "📚 Starting Storybook dev server..."
	npm --prefix frontend run storybook

frontend-storybook-build: .make/frontend-install
	@echo "📦 Building Storybook..."
	npm --prefix frontend run build-storybook

web: frontend-build
	@echo "🌐 Starting web server..."
	@uv run clem web

test: .make/init frontend-test
	@echo "🧪 Running backend tests with coverage..."
	uv run pytest tests/ -v --cov=src/clem --cov-report=term-missing --cov-report=html
	@echo "✓ Tests complete. Coverage report: htmlcov/index.html"

lint: .make/init frontend-lint
	@echo "🔍 Running backend linters..."
	@echo "  → ruff check"
	@uv run ruff check src/clem
	@echo "  → ruff format check"
	@uv run ruff format --check src/clem
	@echo "  → mypy"
	@uv run mypy src/clem
	@echo "✓ All linting checks passed!"

format: .make/init frontend-format
	@echo "🎨 Formatting backend code..."
	@uv run ruff format src/clem
	@uv run ruff check --fix src/clem
	@echo "✓ Code formatted!"

docs:
	make -C docs/diagrams diagrams

.make/tool-install:
	# https://docs.astral.sh/uv/guides/tools/#installing-tools
	uv tool install git+https://github.com/neozenith/clem@main
	@touch $@

run: .make/tool-install .make/init
	@uvx clem

upgrade:
	@uv tool upgrade clem

# Show current memory statistics
stats:
	@echo "Memory Statistics:"
	@uv run $(PWD)/clem.py memory-stats --user $(USER)

# Clean generated artifacts (for testing)
clean:
	@echo "⚠️  This will remove all generated knowledge artifacts!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -f ~/.claude/memories.duckdb; \
		rm -f BDD.md; \
		rm -f ~/.claude/misc/PATTERNS.md; \
		rm -f ~/.claude/misc/LESSONS.md; \
		rm -f ~/.claude/misc/SKILLS.md; \
		rm -rf tmp/; \
		rm -rf ~/.clem/; \
		rm -rf htmlcov/; \
		rm -f .coverage; \
		rm -rf src/clem/web/frontend/; \
		rm -rf frontend/node_modules/; \
		rm -rf .make/; \
		echo "✓ Artifacts cleaned"; \
	else \
		echo "Cancelled"; \
	fi


# npx -p @kayvan/markdown-tree-parser md-tree extract-all lessons.md 2 --output ./.claude/lessons/