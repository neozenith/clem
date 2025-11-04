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
	uv sync --dev
	@touch $@

.make/tool-install:
	# https://docs.astral.sh/uv/guides/tools/#installing-tools
	uv tool install git+https://github.com/neozenith/clem@main
	@touch $@

run: .make/tool-install .make/init
	@uvx clem

upgrade:
	@uv tool upgrade clem
