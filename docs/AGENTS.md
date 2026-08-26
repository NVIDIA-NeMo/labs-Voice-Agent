# Documentation Agent Guidance

## Role

Maintain the user-facing Fern documentation under `docs/` as part of the same change that modifies product
behavior. Verify every command, default, path, protocol contract, and technical claim against checked-in source.
Preserve page paths, heading anchors, local links, code examples, generated navigation, and the repository's
SPDX-first Fern Markdown conventions unless the task explicitly requires a change.

## Writing Style Guide

Apply these rules to documentation, examples, headings, UI text, and release
notes that you create or edit.

- Write in a professional, active, conversational, and engaging voice.
- Use active voice whenever possible. Use present tense for product behavior.
  Address the reader in second person as "you."
- Keep sentences concise. Prefer sentences with fewer than 30 words.
- Use plain English and precise technical terms. Avoid jargon, filler,
  colloquialisms, and flowery marketing claims.
- Avoid contractions in technical documentation. Write "do not," "cannot,"
  and "it is."
- Write "NVIDIA" in all caps and use "an NVIDIA," not "a NVIDIA."
- Spell out uncommon abbreviations on first use. Spell out LLM, RAG, SLM, VLM,
  and MoE on first use.
- Use NVIDIA spellings such as data center, dataset, open source, pretrained,
  startup, webpage, website, and Wi-Fi.
- Replace Latinisms with plain English. Use "for example," "that is," "and so
  on," "through," and "compared to."
- Use "refer to" instead of "see," "can" instead of "may" for possibility,
  and "after" instead of "once" for time.
- Do not use "please" in technical instructions.
- Use numerals for specific values, parameters, measurements, and values of 10
  or more. Spell out zero through nine in general prose.
- Include a space between a number and its unit. Use a comma in numbers with
  four or more digits.
- Use title case for headings. Do not style headings with code, bold, italics,
  quotation marks, ampersands, or exclamation marks.
- Use the Oxford comma. Put periods inside quotation marks in U.S. style.
- Use hyphens only for compound modifiers before nouns. Do not hyphenate an
  adverb that ends in "ly."
- Format commands, code, filenames, paths, and API identifiers as code. Use
  bold for UI elements and the greater-than sign for UI navigation.
- Introduce lists, tables, code examples, and images with a complete sentence.
  Use parallel construction in lists.
- Use descriptive link text. Do not use raw URLs in running text or generic
  link text such as "click here" or "read more."
- Write dates as Month DD, YYYY. Omit the year when it matches the publication
  year. Write time with a 12-hour clock and include minutes only when needed.
- Do not rewrite quoted UI labels, API field names, or audience role labels in
  tables to enforce second person.
- Provide useful alt text and preserve a logical heading hierarchy.
- Verify commands, flags, API names, defaults, and technical claims against
  source code or another checked-in source of truth.
- Do not rewrite literal code, identifiers, commands, URLs, or quoted terminal
  and API output to satisfy prose rules.
- Apply rules to improve clarity. Do not make mechanical changes that reduce
  technical accuracy or readability.

## NVIDIA DORI Routing

1. Check whether the current agent exposes `dori_handle` or `dori_route` and
   `dori_collections`.
   If the user explicitly asks not to use DORI, use the
   [Writing Style Guide](#writing-style-guide) instead.
2. When those tools are available, list the installed collections.
   - If a collection source exactly matches `gitlab:tech-docs/skill-library`,
     use DORI for task routing.
   - If the collection is missing, inaccessible, or cannot be verified,
     continue with the [Writing Style Guide](#writing-style-guide).
3. When the DORI tools are unavailable, continue with the Writing Style Guide.
   Do not inspect a shell-visible CLI, install software, or configure the host
   during a normal documentation task.

Capability detection does not approve installation or host configuration.
DORI unavailability must not block documentation work. Do not add DORI
installation, private-source access, or host-configuration instructions to this
open-source repository.

Use the following DORI workflow only when current host capabilities include the
verified NVIDIA documentation Skill Library. Complete the documentation before
the developer opens the pull request.

1. Route the documentation task through DORI. Include the changed source files,
   the user-visible impact, the documentation that might need updates, and the
   required validation.
2. Follow the skill or workflow that DORI returns. Verify product behavior
   against checked-in sources before drafting.
3. When the host supports subagents, start a documentation subagent while the
   primary developer finishes the implementation. Reconcile the documentation
   changes and validation evidence before opening the pull request.
4. When the host does not support subagents, complete the same documentation
   work in the primary task.

If the verified Skill Library is unavailable, inaccessible, or fails, skip DORI.
Do not attempt routing or start setup automatically. Continue using the Writing
Style Guide above.

## Documentation Impact Map

Check the matching page whenever you change the following source surfaces:

| Changed Source | Documentation to Check |
|---|---|
| A config key, default, or `server_configs/**` YAML | `docs/build-voice-agents/configure/`, `docs/reference/runtime/config-schema.md` |
| A `build_*` function or the pipeline order | `docs/about/architecture.md`, `docs/build-voice-agents/extend/pipelines/builders.md` |
| An STT, TTS, diarization, or turn-taking service or model | The matching `docs/about/core-concepts/speech-pipeline/*.md` page |
| LLM backend selection, reasoning, or omni | `docs/about/core-concepts/language-models/{llm,reasoning,multimodal}.md` |
| vLLM serving, plugins, or hosted NVIDIA endpoints | `docs/build-voice-agents/model-serving/{vllm,vllm-plugins,nvidia-nim}.md` |
| Tool calling or `utils/tool_calling/**` | `docs/build-voice-agents/tools/{tool-calling,custom-tools}.md` |
| An RTVI action or the `/connect` handshake | `docs/build-voice-agents/extend/protocols/{rtvi-actions,client-protocol}.md`, `docs/reference/runtime/rtvi-messages.md` |
| `run_evaluation.py` flags, defaults, or scoring | `docs/evaluate/understand-scoring/scoring.md`, `docs/evaluate/run-evaluations/resume.md`, `docs/reference/evaluation/eval-cli.md` |
| Evaluation scenarios, tools, or fixtures for a domain | `docs/evaluate/domain-guides/*.md`, `docs/evaluate/create-evaluations/authoring-*.md` |
| A metric written to `metrics.json` or `all_summary.txt` | `docs/reference/evaluation/metrics.md` |
| An environment variable | `docs/reference/runtime/environment.md` |
| Dependencies, Python versions, test layout, or lint tooling | `docs/resources/contribute/{index,testing}.md` |

Apply these source-of-truth rules:

- **Documentation in code counts.** An argparse `help=` string, config comment, or docstring is documentation.
  Fix it in the same change. A help string that contradicts its default is a common defect in this repository.
- **Verify instead of copying.** Do not restate a claim from `README.md` or `AGENTS.md` without checking the
  source. Cite the file that you opened.
- **Use consistency tests as a backstop.** `uv run pytest tests/unit/test_docs_consistency.py` checks mechanical
  contracts such as enum counts, CLI defaults, and referenced paths. It does not prove behavioral prose correct.
- If a change makes a page inaccurate and you cannot fix it in scope, report the gap instead of leaving it
  silently stale.

## Documentation Site

`docs/` is a Fern site published to `docs.nvidia.com/nemo/labs-voice-agent` through `docs/fern/docs.yml`.
Merging a `docs/**` change to `main` publishes it when the `PUBLISH_FERN` repository variable is enabled.
Review the pull request preview because the repository has no staging channel.

The documentation workflows are `fern-docs-ci.yml`, `fern-docs-preview-build.yml`,
`fern-docs-preview-comment.yml`, and `publish-fern-docs.yml` under `.github/workflows/`.

Follow these maintenance rules:

- **Generate navigation.** `docs/fern/nav.json` is the source for `docs/fern/versions/nightly.yml` and
  `docs/index.yml`. Do not hand-edit the generated files. Edit `nav.json`, run
  `npm --prefix docs/fern run nav:gen`, and verify with `npm --prefix docs/fern run nav:check`.
- **Preserve Fern Markdown safety.** Author pages as `.md`; `.mdx` is generated only. Fern renders `.md`
  through MDX, so bare braces and angle brackets outside code fences can break the build. Authored pages use
  an SPDX JSX comment followed by one body H1.
- **Keep local links resolvable.** Fern CI runs offline link validation over `docs/**/*.md`, so every relative
  target must exist on disk.
- **Treat API reference pages as generated.** `docs/fern/product-docs/**` is gitignored. A prose change needs
  `npm --prefix docs/fern run check`; a local rendered preview also needs the documented library-generation
  workflow in `docs/fern/README.md`.

## Validation

Run validation in proportion to the documentation change. At minimum, use the following commands for affected
surfaces:

```bash
uv run pytest tests/unit/test_docs_consistency.py
npm --prefix docs/fern run nav:check
npm --prefix docs/fern run check
git diff --check
```

For changes to the documentation review receipt, also run:

```bash
uv run pytest tests/unit/test_docs_review_receipt.py
python scripts/docs-review-receipt.py --help
```

Review the generated pull request preview for navigation, layout, code blocks, tables, and links before merge.
