
> This project originally started as a standalone script and a sibling test file and a variety of markdown files capturing project requirements and designs.\
  \
  I have now copied over those files into this new repo so that it can expand in complexity beyond a single file script and so that the test suite can be
  comprehensive to make sure the parts of the system are small, well reasoned and interact coherently, cohesively and with low coupling.\
  \
  After some reflection I want to update the goals of the project and I want you to assist with this refactoring and restructuring. Some of these updated may align
  with existing goals and contradict others. Consider this new list as the cannonical list of goals going forward. The updated goals:\
  \
  - INPUTS: The session logs per Project for claude code in ~/.claude/projects/**/*.jsonl where each file IS a `Session` that is getting updated with `Events`
  everytime there is a `User Message`, `Assistant Message`, `Tool Call` etc and we are to leverage `duckdb` to beable to read from this source for live details\
  - DOMAINS: A `Project` is defined as a single git repo that Claude Code works with. A `Project` could have many `Sessions` which account for a lot of `Events` in
  how the user and Claude Code are evolving a code `Project`. A `Domain` is a collection of `Projects`. For example `~/play/` has all of my personal git projects
  which has it's own style, quality and design decisions. `~/work/` is the `Domain` of internal work projects and Proofs of Concept (POCs) and rapid prototypes as
  well as templates that would act as accelerators of IP prior knowledge. I also work in consulting and may work on various clients and they have a collection of
  repos like `~/clients/acmeinc/` would have all the Projects for `acmeinc` where as `~/clients/brandname/` would be all the repos for `brandname`. Each of these
  represent domains that have their own preferences and styles. Lastly, sometimes I will grab the source code of opensource projects and put them in `~/foss/` and
  then use claude code for exploratory purposes to analyse the code and sometimes make changes to contribute improvements back. So segregation of Domains is
  important in the hierarchy of retrieving and distilling knowledge.\
  - USAGE: `clem` should be installable via `uv tool install git+https://github.com/neozenith/clem@main` and then the cli can be run via `uvx clem`\
  - OBSERVABILITY: `uvx clem web` should start a local web server which will start a locally run FastAPI webserver which will host our frontend webapp. It will be
  a plain React application built with Vite that interacts the FastAPI webserver which holds the logic for interacting with the knowledge base we will be curating.
  This will become a sophisticated visual into CLEM but for now it should provide a way of visualising the Sessions and Events per Project and Domain\
  - RETRIEVAL: I want CLEM to be able to run as an MCP for Claude Code. Claude Code should be able retrieve knowledge at Project Level, Domain Level and Global
  Level\
  - REQUIREMENTS EXTRACTION: During the course of a Session on a Project a User will often implicitly specify requirements about a Project like "It should do X".
  These should get extracted and added to the CLAUDE.md\
  - INCREMENTAL LEARNING: The only real way for Claude Code to get better is allowing Claude Code to update the `.claude` folder on a Project. This includes
  `skills`, `plugins`, `agents`, `commands` etc. This incremental learning mechanism will need to check the live documentation of Claude Code. I imagine this would
  work by creating a slash command for claude code that would leverage the MCP for latest insights and then prompt claude code with tactical changes that get made
  to that project's .claude folder. I will often setup the .claude folder as a version controlled repo shared across Domains and Projects. These changes should be
  scoped to Globally appropriately abstracted improvements. Additionally to the builtin features of Claude Code I also want to curate my own directory of
  `.claude/lessons/` which is currently curated under `.claude/misc` which has markdown documents that have proven incredibly useful for priming a Claude Code
  Session context\
  - MEMORY: The internals of this I imagine may change and elaborate over time. We should start with Agentic Memory extraction and consolidation using huggingface
  models. We should also form base level knowledge graphs. We should also create hierarchical layers of community detection on the knowledge graph. We should use
  GraphRAG which also needs Node2Vec to be able to search a graph across a vector space. I want all of this to be managed and maintained in `~/.clem/memory.duckdb`
  which will leverage the Vector Similarity Search (VSS) of DuckDB and create it's own schemas for managing the knowledge graph in the duckdb database