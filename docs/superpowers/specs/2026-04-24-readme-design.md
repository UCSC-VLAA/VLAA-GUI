# README Rewrite Design

## Goal

Rewrite `README.md` so the repository reads like an academic paper release rather than product-style framework documentation, while still keeping installation and usage practical for researchers.

## Approved Direction

Use a balanced academic structure:

- `VLAA-GUI` remains the primary title.
- The paper title appears as a subtitle.
- Research framing comes first, but setup and usage remain easy to find.

## Content Plan

1. Add a paper-style hero section with logo, title, authors, affiliations, and links.
2. Mark the paper link as `arXiv coming soon`.
3. Lead with research-facing content:
   - overview paragraph adapted from the paper abstract
   - main results figure from `assets/results.png`
   - method overview from `assets/pipeline.png`
   - concise benchmark highlights
4. Keep an academic-style repository overview:
   - code structure
   - installation
   - configuration
   - local interactive usage
   - OSWorld evaluation notes
   - WindowsAgentArena note only if supported by actual repo contents
5. End with citation placeholder and acknowledgements.

## Accuracy Constraints

- Pull paper title, authors, affiliations, and benchmark numbers from `/Users/sergiu/mypapers/eccv2026/main.tex`.
- Only reference figures that exist under `assets/`.
- Remove or avoid README links to files that do not exist in this checkout, especially `waa_setup/README.md`.
- Keep commands aligned with real repo entry points such as `uv sync`, `uv run agent --config-path ...`, and `scripts/run_agent.sh`.

## Writing Style

- Academic tone, not marketing tone.
- Concise, skimmable sections.
- Research claims should mirror the paper language closely enough to stay accurate, but be shorter than the abstract.
- Operational sections should be direct and minimal.
