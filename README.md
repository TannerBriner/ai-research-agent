# Autonomous Research Agent

Give it a topic, and it decides for itself how to research it: it plans
sub-questions, searches the web as many times as it judges necessary,
adapts its next query based on what it just learned, and writes a
structured, numbered-citation report once it decides it has enough
information — no fixed number of steps, no hand-written pipeline telling
it what to do next.

**Tech:** Python · Anthropic Claude API (native tool-use / function-calling) · Tavily Search API

## Problem

Most "AI-powered search" demos are really a single retrieval step
followed by one summarization call — useful, but not actually agentic.
This project is a genuine agentic loop: the model itself holds the
`web_search` tool and decides, turn by turn, whether it needs to search
again, what to search for next, or whether it's ready to write the final
report. The goal was to build and understand that pattern — plan, act,
observe, decide, repeat — rather than to solve a specific business
problem, as a hands-on demonstration of working with LLM tool-use APIs.

## Approach

- **Agent loop** (`run_research_agent()`): each turn, Claude reads the
  conversation so far and either calls `web_search` again or returns a
  final answer. The loop only ends when the model's `stop_reason` is no
  longer `tool_use` — the model decides when it's done, capped at 15 turns
  as a safety valve against runaway searching.
- **Tool definition**: `web_search` is described to Claude via the
  Messages API's tool schema (name, description, JSON Schema input).
  Claude never executes anything itself — it requests a query, and the
  script runs the actual Tavily search and hands the results back as a
  `tool_result` message.
- **Citation handling, split between prompt and code on purpose**: the
  system prompt asks Claude to cite sources inline as normal markdown
  links (`[Label](URL)`) while writing, since that's the format LLMs
  produce most reliably mid-generation. A Python post-processing pass
  (`convert_citations()`) then converts those into numbered, MLA-style
  citations with a matching Works Cited list. Numbering is deliberately
  done in code, not by the model — keeping 15-30 citation numbers
  self-consistent across a long generation is a common LLM failure point,
  while string substitution over text you already have is deterministic.
- **Defensive post-processing**: real test runs surfaced several ways
  models drift from an instructed citation format — wrapping an entire
  sentence in the link instead of a short label, dropping bare
  `[SourceName]` tags with no URL attached, gluing a source name directly
  onto a sentence with no punctuation. Rather than trust prompting alone
  to prevent every variant, the code has explicit fallbacks for each:
  `resolve_orphan_tags()` reconnects bare tags to their source by name
  match, and every visible citation label is auto-wrapped in parentheses
  regardless of what the model wrote, so the output is grammatically
  consistent either way.
- **Output cleanup**: `strip_preamble()` removes any conversational
  lead-in the model adds despite being told not to; the model's own
  closing "Sources" section is discarded and replaced with a
  Python-generated Works Cited list built from the same URLs used for
  inline numbering, so the two can never drift out of sync.

## Results

Four example reports in `example_reports/`, chosen to cover a range of
research shapes: a lighthearted factual lookup (favorite dog breed), a
topic with genuinely conflicting evidence across sources (does remote
work hurt productivity), a fast-moving technical/current-events topic
(state of AI agent frameworks in production), and a health-and-nutrition
question with real nuance (air frying vs. deep frying). Across these
runs the agent typically issues 4-6 searches per topic and cites roughly
10-30 distinct sources per report, depending on how much the topic
branches into sub-questions.

## What I'd do next

- Add a lightweight source-quality signal (e.g. deprioritizing SEO
  content farms in favor of primary sources) rather than treating every
  search result as equally citable
- Cache/rate-limit Tavily calls so repeated runs on similar topics don't
  re-search from scratch
- Add automated tests for the citation regex pipeline instead of relying
  on manual review of each generated report
- Stream the agent's reasoning/search activity to the terminal in real
  time instead of only printing each search query as it happens

## How to run this

```bash
git clone https://github.com/TannerBriner/ai-research-agent.git
cd ai-research-agent
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and TAVILY_API_KEY
python src/research_agent.py "Is air frying actually healthier than deep frying?"
```

## Repo structure

```
src/research_agent.py   — the agent loop, tool definition, and citation post-processing
example_reports/         — sample generated reports (four topics, see Results above)
.env.example             — template for required API keys
requirements.txt         — anthropic, tavily-python, python-dotenv
```
