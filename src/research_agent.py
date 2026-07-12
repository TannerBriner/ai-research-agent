"""
Autonomous Research Agent

Given a topic, this agent autonomously:
  1. Breaks it into concrete sub-questions worth investigating
  2. Searches the web (Tavily) as many times as it decides it needs to,
     refining queries based on what it learns
  3. Synthesizes a structured, cited markdown report once it judges it
     has enough information

This is a genuine agentic loop, not a fixed pipeline: the model itself
decides what to search, how many times, and when it's done, using
Claude's native tool-use API. That's the difference between this and a
simple "search then summarize" script.

Citation handling: the model is asked to cite sources inline as normal
markdown links ([Title](URL)) while it writes, because that's the format
LLMs are most reliable at producing consistently. This script then
post-processes that output in Python to convert it into numbered,
MLA-flavored citations (e.g. "...as reported [3]...", with a matching
numbered Works Cited list at the end). Numbering is done in code rather
than by the model, because keeping 15-20 citation numbers self-consistent
across a long generation is a common failure point for LLMs -- string
matching in Python is guaranteed correct every time.
"""

import os          # for reading environment variables (API keys, model override)
import re          # for the citation-finding/rewriting regex in convert_citations()
import sys         # for sys.exit() when required API keys are missing
import argparse    # for parsing the command-line topic/output arguments
from datetime import date          # for the "Generated on <date>" line in the saved file
from urllib.parse import urlparse  # for pulling a short site name (e.g. "akc.org") out of each URL

from dotenv import load_dotenv   # reads key=value pairs out of a local .env file into os.environ
import anthropic                 # Anthropic's official Python SDK -- talks to the Claude API
from tavily import TavilyClient  # Tavily's official Python SDK -- talks to the Tavily search API

# load_dotenv() reads the .env file (if present) in the current directory and
# copies its key=value pairs into os.environ, so the os.environ.get() calls
# just below can find them. This keeps real secrets out of the source code.
load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Verify the current model ID at https://docs.anthropic.com/en/docs/about-claude/models
# before running -- model slugs change over time and this default may be stale.
# ANTHROPIC_MODEL can be set in .env to override this without touching code.
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# Safety valve on the agent loop below: caps how many back-and-forth
# "search, read results, decide what to do next" cycles the agent can run
# for a single topic, so a stubborn/looping model can't run forever (and
# rack up API costs) on one request.
MAX_AGENT_TURNS = 15

# This system prompt is the agent's entire "job description." It tells the
# model both what to do (research, using the tool) and exactly how its
# final output should be formatted, so the rest of this script can
# reliably post-process that output (see convert_citations() and
# strip_preamble() below).
SYSTEM_PROMPT = """You are an autonomous research agent. Given a research topic, your job is to:

1. Break it into 2-4 concrete sub-questions worth investigating.
2. Use the web_search tool to gather information for each sub-question.
   Call it as many times as you need, refining your queries based on what
   you learn. Do not stop after a single search if the topic has multiple
   facets -- build a well-rounded picture from multiple sources first.
3. Once you have enough information, synthesize a well-organized markdown
   report.

Your final report MUST:
- Have a clear structure with headers for each sub-topic
- Cite sources inline using [Source Title](URL) format
- End with a "Sources" section listing every URL you actually used
- Explicitly note any conflicting information or gaps you found, rather
  than glossing over them

Do not fabricate sources or facts. Only cite URLs that were actually
returned by the web_search tool.

IMPORTANT citation style: the bracketed link text must be a SHORT source
label only -- 1 to 3 words, naming ONLY the publication, organization, or
outlet (e.g. "AKC", "CBS News", "ScienceDirect") -- never the factual
claim itself, never a paraphrase of it, and never a descriptive clause
about what the source says. If you find yourself writing more than three
words inside the brackets, or writing anything that describes content
rather than naming a source, stop and shorten it to just the outlet name.
Write the sentence as plain text first, then attach the short citation
immediately after it.

Correct:   The French Bulldog has held the top spot for four consecutive
           years [AKC](https://www.akc.org/most-popular-breeds).
Incorrect: [The French Bulldog has held the top spot for four consecutive
           years](https://www.akc.org/most-popular-breeds).
Also incorrect: The French Bulldog has held the top spot for four
           consecutive years [AKC].
Also incorrect: LangGraph is designed for building stateful, multi-actor
           applications [LangGraph is designed specifically for building
           stateful, multi-actor applications powered by
           LLMs](https://www.turing.com/resources/ai-agent-frameworks).

The first incorrect style turns your entire sentence into a hyperlink.
The second incorrect style -- citing with a bare bracketed name and no
URL attached to it -- is just as bad: every single inline citation, with
NO exceptions, must have its URL attached directly in the same
[Label](URL) pair, every time you cite it, even if you cited that same
source earlier in the report. Never rely on a separate source list alone
to carry the URL. The third incorrect style repeats the claim a second
time inside the brackets instead of naming the source -- the sentence
already states the fact; the bracket's only job is to name WHO said it,
in as few words as possible.

Your final response must contain ONLY the report itself, starting
directly with its title as a top-level markdown heading (# Title). Do not
include any conversational preamble, meta-commentary, or acknowledgment
before it (for example, do not say things like "Now I have enough
information, let me write the report" -- go straight to the "#" heading).
"""


def web_search_tool_definition():
    """
    Describes the web_search tool to Claude in the exact schema the
    Messages API's tool-use feature expects: a name, a plain-English
    description (the model reads this to decide WHEN to call the tool),
    and a JSON Schema for its input (here, just one required string field,
    "query"). Claude never actually executes this function itself -- it
    just decides to call it and supplies the "query" argument; our code
    below is what actually runs the search and hands the results back.
    """
    return {
        "name": "web_search",
        "description": (
            "Search the web for current information on a topic. Returns a "
            "list of results, each with a title, URL, and content snippet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run",
                }
            },
            "required": ["query"],
        },
    }


def run_web_search(tavily_client, query, max_results=5):
    """Actually calls the Tavily search API for one query and returns the
    raw list of result dicts (each with title/url/content keys)."""
    response = tavily_client.search(query=query, max_results=max_results)
    return response.get("results", [])


def format_search_results_for_claude(results):
    """
    Converts Tavily's raw result list into a single plain-text block that
    gets fed back to Claude as the tool's output. Claude only ever sees
    this formatted text, not the original Python dicts, so the model can
    read titles/URLs/content the same way it would read any other text.
    """
    if not results:
        return "No results found for this query."
    chunks = []
    for r in results:
        chunks.append(
            f"Title: {r.get('title')}\n"
            f"URL: {r.get('url')}\n"
            f"Content: {r.get('content')}"
        )
    # "---" separators make it visually obvious to the model where one
    # search result ends and the next begins.
    return "\n\n---\n\n".join(chunks)


def run_research_agent(topic, max_turns=MAX_AGENT_TURNS, verbose=True):
    """
    Runs the actual agent loop for one topic. This is the core of the
    project: rather than a fixed "search once, then summarize" pipeline,
    Claude is given the web_search tool and allowed to call it repeatedly,
    reasoning about what it's learned so far and deciding what to search
    next -- exactly like a human researcher would -- until it decides it
    has enough information and writes the final report instead of calling
    the tool again.
    """
    # Two separate API clients: one for the LLM (Anthropic), one for web
    # search (Tavily). The agent loop below is what ties them together.
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

    # `messages` is the running conversation history sent to Claude on
    # every turn. It starts with just the user's research request, and we
    # append to it as the loop progresses (Claude's responses, and our
    # tool results) so each new API call has full context of everything
    # that happened before it.
    messages = [
        {
            "role": "user",
            "content": f"Research this topic and produce a cited report: {topic}",
        }
    ]
    # Tracks every source Claude actually searched and saw, across every
    # turn of the loop -- used only for the "Sources consulted" count
    # printed at the end (the report's own citations are handled
    # separately by convert_citations(), based on what Claude actually
    # cited rather than everything it merely saw).
    sources_used = []

    for turn in range(max_turns):
        # Ask Claude to continue: read the conversation so far and either
        # (a) call the web_search tool again, or (b) write the final
        # report. Which one happens is entirely the model's decision --
        # that's what makes this "agentic" rather than scripted.
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            tools=[web_search_tool_definition()],
            messages=messages,
        )

        # Record Claude's response (whatever it was) into the running
        # conversation history before we look at what it decided to do.
        messages.append({"role": "assistant", "content": response.content})

        # response.stop_reason tells us WHY Claude stopped generating.
        # "tool_use" means it wants to call web_search again; anything
        # else (usually "end_turn") means it decided it's done and this
        # response is the final report -- so we extract the plain text
        # and return, ending the loop.
        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return final_text, sources_used

        # Otherwise, Claude's response contains one or more tool_use
        # blocks (it can request multiple searches in a single turn).
        # We run each requested search for real, then package the results
        # up as "tool_result" messages so Claude can read them on its next
        # turn -- this is the hand-off point between "the model decided
        # what to search" and "our code actually went and searched it."
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "web_search":
                query = block.input["query"]
                if verbose:
                    print(f"  [turn {turn + 1}] searching: {query!r}")
                results = run_web_search(tavily_client, query)
                for r in results:
                    sources_used.append(
                        {"title": r.get("title"), "url": r.get("url")}
                    )
                tool_results.append(
                    {
                        "type": "tool_result",
                        # tool_use_id links this result back to the specific
                        # tool call it's answering -- required by the API
                        # when a single turn contains multiple tool calls.
                        "tool_use_id": block.id,
                        "content": format_search_results_for_claude(results),
                    }
                )
        # Tool results are sent back as a "user" turn (from the API's
        # perspective, tool output is information arriving from outside
        # the model, same as a user message would be).
        messages.append({"role": "user", "content": tool_results})

    # If we fall out of the for-loop, Claude used up every allowed turn
    # without ever stopping to write a final report -- most likely on an
    # overly broad topic. Return a clear message instead of crashing.
    return (
        "Agent did not converge to a final report within the turn limit. "
        "Consider raising max_turns or narrowing the topic.",
        sources_used,
    )


def slugify(topic):
    """Turns a topic string into a safe filename fragment, e.g.
    'Americas favorite dog breed?' -> 'americas_favorite_dog_breed'.
    Keeps only letters/digits/spaces, lowercases everything, then joins
    words with underscores and caps the length so filenames stay
    reasonable."""
    keep = "".join(c if c.isalnum() or c == " " else "" for c in topic)
    return "_".join(keep.lower().split())[:60]


def strip_preamble(text):
    """
    Defensive cleanup: even with the system prompt's explicit instruction,
    the model can occasionally slip in a line of conversational narration
    before the report itself (e.g. "Great, now I have enough information,
    let me write this up."). This finds the first line that looks like a
    markdown heading ("# ...") and discards everything before it, so the
    saved file always starts cleanly with the model's own title -- a
    safety net in case the prompt instruction alone isn't followed.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            return "\n".join(lines[i:]).strip()
    # Fallback: no heading found anywhere (shouldn't normally happen) --
    # return the text as-is rather than silently deleting everything.
    return text.strip()


def domain_from_url(url):
    """Extracts a short, readable site name from a URL for use in the
    Works Cited list, e.g. 'https://www.akc.org/most-popular-breeds'
    -> 'akc.org'. Strips a leading 'www.' since it's just visual noise."""
    netloc = urlparse(url).netloc
    return netloc[4:] if netloc.startswith("www.") else netloc


def resolve_orphan_tags(body, order):
    """
    Fallback safety net for a second citation failure mode, distinct from
    the one convert_citations() already guards against.

    Sometimes the model skips the [Label](URL) format entirely partway
    through a report and instead drops a bare citation marker straight
    into a sentence, e.g. "...growing at 43.84% CAGR [Landbase]" -- a
    source name in brackets with no "(url)" attached at all. Because
    there's no URL right after it, convert_citations()'s main
    link_pattern never matches these, so they'd otherwise survive
    untouched in the final report as dead-end references: a reader sees
    "[Landbase]" but has no way to trace it to a specific numbered entry
    in the Works Cited list.

    This function catches those leftover bare tags and reconnects them:
    it looks up each tag's text against the source titles we already
    collected in `order` (built earlier in convert_citations() from every
    proper [Label](URL) pair found anywhere in the report, including the
    model's own closing "Sources" section) and, on a match, rewrites the
    bare tag as the same "[n]" number used for that source everywhere
    else -- so the inline mention and the Works Cited entry always agree.

    Args:
        body: report text after the main [Label](URL) substitution and
            after the "## Sources" section has been stripped out.
        order: list of (number, title, url) tuples, in first-seen order,
            built by convert_citations() from every [Label](URL) match.

    Returns:
        The body text with resolvable bare tags rewritten as "[n]", and
        any unresolvable ones reduced to plain text (bracket-stripped)
        rather than left as a dangling, meaningless reference.
    """
    # Lowercase title -> citation number, so the lookup below is not
    # case-sensitive (the model may write "landbase" inline but "Landbase"
    # in its Sources section, or vice versa).
    title_to_number = {title.lower(): number for number, title, url in order}

    # Matches a bracketed tag that STARTS WITH A LETTER -- this is what
    # keeps this pattern from ever re-matching citation numbers we've
    # already inserted, like "[1]" or "[12]", which start with a digit.
    # The negative lookahead "(?!\()" excludes anything immediately
    # followed by an opening paren, which would mean it's actually a
    # proper [Label](URL) pair that the main pattern already handled
    # (or, in rare cases, should be left for that pattern to handle).
    orphan_pattern = re.compile(r"\[([A-Za-z][^\]]*)\](?!\()")

    def replace_orphan(match):
        tag_text = match.group(1)
        number = title_to_number.get(tag_text.lower())
        if number is not None:
            # Matched a known source -- rewrite the bare tag as our
            # proper numbered citation, now in sync with Works Cited.
            return f"[{number}]"
        # No matching source was ever collected for this tag (the model
        # named a source but never gave us a URL for it anywhere). Rather
        # than leave a bracketed reference that points nowhere, strip the
        # brackets and keep the bare word(s) as plain text -- no content
        # is deleted, it just stops looking like a citation.
        return tag_text

    return orphan_pattern.sub(replace_orphan, body)


def convert_citations(report_text):
    """
    Rewrites the model's inline markdown-link citations into numbered,
    MLA-flavored citations, and appends a matching numbered Works Cited
    list.

    Why this is done in Python instead of asking the model to number its
    own citations: an LLM writing a long report with 15-20 citations is
    prone to renumbering drift (e.g. reusing [3] for two different
    sources, or skipping a number). Regex substitution over text we
    already have full control of is deterministic and always correct.

    How it works:
      1. Find every occurrence of the model's [Title](URL) markdown link
         pattern in the report body.
      2. The first time a given URL is seen, assign it the next citation
         number and remember it; if the same URL is cited again later in
         the report, reuse its existing number instead of creating a
         duplicate entry.
      3. Replace each inline link with its ORIGINAL visible text plus a
         bracketed number, e.g. [Title](URL) -> "Title [1]". The visible
         text is deliberately kept rather than discarded: the system
         prompt asks the model to only ever link a short source label
         (like "AKC"), but if it ever ignores that and wraps a whole
         sentence in the link instead, keeping the text guarantees the
         report's actual content is never silently deleted -- worst case
         a citation number lands after a full sentence instead of a short
         tag, which is a cosmetic issue rather than a content bug.
      4. Drop whatever "## Sources" section the model wrote (its links are
         now just plain numbers with nothing to point a reader to), and
         replace it with our own numbered Works Cited list, generated
         directly from the same URLs collected in step 2 -- guaranteeing
         the numbers always match between the inline citations and the
         list at the bottom.
    """
    # Matches markdown links like [Some Title](https://example.com/page) --
    # group 1 captures the link text, group 2 captures the URL.
    link_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)")

    seen_urls = {}   # url -> citation number already assigned to it
    order = []       # (number, title, url) tuples in first-seen order

    def replace_link(match):
        title, url = match.group(1), match.group(2)
        if url not in seen_urls:
            # First time this URL has been cited: assign it the next
            # available number and record it for the Works Cited list.
            number = len(order) + 1
            seen_urls[url] = number
            order.append((number, title, url))
        else:
            # Already cited earlier in the report: reuse the same number
            # instead of creating a second entry for the same source.
            number = seen_urls[url]
        # Keep `title` as plain visible text and append the citation
        # number after it -- never discard it. If the model followed the
        # system prompt's instruction, `title` is just a short tag like
        # "AKC" and this reads as "AKC [1]". If the model ignored that
        # instruction and wrapped a full sentence instead, this still
        # preserves the whole sentence and simply appends "[1]" after
        # it -- either way, no report content is ever lost.
        #
        # Always wrap the visible label in parentheses before the number,
        # e.g. "AKC [1]" -> "(AKC) [1]". This is deliberately done here in
        # code rather than left to the model: without it, a bare source
        # name gets glued directly onto the sentence with no punctuation
        # marking it as a citation, reading like a typo -- e.g. "...running
        # in production Punku AI [1]" instead of the intended "...running
        # in production (Punku AI) [1]". Checking for existing parentheses
        # first avoids double-wrapping on the rare occasion the model
        # already added its own, e.g. "(BLS)" -> stays "(BLS)", not "((BLS))".
        display_title = title.strip()
        if not (display_title.startswith("(") and display_title.endswith(")")):
            display_title = f"({display_title})"
        return f"{display_title} [{number}]"

    # re.sub walks the whole report and calls replace_link() on every
    # match, building the numbered-citation version of the text in one pass.
    body = link_pattern.sub(replace_link, report_text)

    # The model was also asked to write its own "## Sources" section full
    # of the same markdown links -- those links have now been turned into
    # bare numbers by the substitution above, so that section is no longer
    # useful. re.split on the "## Sources" (or "# Sources") heading and
    # keeping only the text BEFORE it removes that now-broken section,
    # leaving the rest of the report body intact.
    body = re.split(r"\n#{1,2}\s*Sources\b.*", body, flags=re.IGNORECASE | re.DOTALL)[0].rstrip()

    # Second safety net, for a different failure mode than the one above.
    # Sometimes the model skips the [Label](URL) format entirely and just
    # drops a bare citation marker like "[Landbase]" into the sentence --
    # a source name in brackets with no URL attached at all. Those never
    # matched link_pattern (there's no "(url)" right after them), so they
    # pass through untouched and would otherwise sit in the final report
    # as dead-end references that don't point to anything. resolve_orphan_tags()
    # catches these and reconnects them to the same numbering used above.
    body = resolve_orphan_tags(body, order)

    # Build our own replacement, in citation order, using the exact same
    # numbers assigned above -- this is what guarantees inline [3] and
    # Works Cited entry 3 always refer to the same source.
    works_cited_lines = ["## Works Cited\n"]
    for number, title, url in order:
        works_cited_lines.append(f'{number}. "{title}." {domain_from_url(url)}, {url}.')

    return body + "\n\n" + "\n".join(works_cited_lines) + "\n"


def main():
    # argparse turns command-line arguments into a simple object
    # (args.topic, args.output) instead of us having to parse sys.argv by
    # hand, and it auto-generates --help text for free.
    parser = argparse.ArgumentParser(description="Autonomous research agent")
    parser.add_argument("topic", help="Research topic or question")
    parser.add_argument(
        "--output", "-o", help="Output markdown file path (default: auto-named)"
    )
    args = parser.parse_args()

    # Fail fast with a clear message rather than letting the Anthropic/
    # Tavily SDKs raise a confusing authentication error deeper in the code.
    if not ANTHROPIC_API_KEY or not TAVILY_API_KEY:
        print(
            "Error: set ANTHROPIC_API_KEY and TAVILY_API_KEY in a .env file "
            "(see .env.example)."
        )
        sys.exit(1)

    print(f"Researching: {args.topic}\n")

    # This is where the actual agent loop (defined above) runs to completion.
    report, sources = run_research_agent(args.topic)

    # Post-processing pipeline, in order: strip any leftover conversational
    # preamble, then renumber citations and attach the Works Cited list.
    report = strip_preamble(report)
    report = convert_citations(report)

    # If the user didn't specify --output, build a filename automatically
    # from the topic so every run produces a sensibly-named file without
    # extra effort.
    output_path = args.output or f"example_reports/{slugify(args.topic)}.md"

    # A short metadata line at the top of the file records when it was
    # generated and what the original query was -- useful context for
    # anyone (including future you) reading the report later.
    metadata = (
        f"*Generated {date.today().isoformat()} by the autonomous research "
        f"agent · query: \"{args.topic}\"*\n\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(metadata + report)

    print(f"\nReport saved to {output_path}")
    print(f"Sources consulted across the session: {len(sources)}")


if __name__ == "__main__":
    main()
