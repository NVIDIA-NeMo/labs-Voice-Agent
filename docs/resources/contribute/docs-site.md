{/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/}

# Building the Docs

This site is built with [Fern](https://buildwithfern.com). Page content lives in `docs/` as
Markdown; the Fern configuration lives in `docs/fern/`. Everything below runs from `docs/fern`
(or from the repo root with `npm --prefix docs/fern run <script>`).

## Prerequisites

| Requirement | Why |
|---|---|
| Node.js 20 or newer, installed via a version manager | `docs/fern/package.json` declares `engines.node >= 20`, and CI pins Node 20, so that is the supported floor for `npm run check`. Node 18 fails immediately with `ReferenceError: crypto is not defined`. For `npm run dev` specifically, prefer Node 22.13+: the dev server bootstraps `pnpm`, whose current release declares `node >= 22.13`. Install through `nvm` (or similar) rather than a system package — the dev server installs `pnpm` globally, which fails with `EACCES` when npm's global prefix is a root-owned directory such as `/usr/local`. |
| Docker | Only for `npm run generate:library:local`, which parses the Python package to build the API reference. Nothing else needs it. |
| A Fern account with NVIDIA org access | Only for the dev server, hosted previews, and publish. `docs.yml` sets `global-theme: nvidia`, which the CLI fetches at serve time. |

The Fern CLI is **not** installed globally. Every npm script fetches and runs the exact version
pinned in `docs/fern/fern.config.json` via `npm exec`, so local runs match CI.

## Scripts

| Command | What it does |
|---|---|
| `npm run nav:gen` | Regenerates both navigation files from `nav.json`. Run after any page add/remove/rename. |
| `npm run nav:check` | Exits non-zero if the generated nav files are stale. |
| `npm run check` | `fern check` — validates `docs.yml` and the nightly navigation. No Docker, no login, no generated API reference needed. |
| `npm run login` | Authenticates the CLI so it can fetch the NVIDIA theme. |
| `npm run dev` | Local dev server at `http://localhost:3000`. |
| `npm run generate:library:local` | Builds the Python API reference from the local checkout (Docker). |
| `npm run generate:library` | Builds the API reference from the GitHub source Fern clones. Requires login. |
| `npm run preview` | Hosted preview build. Needs `FERN_TOKEN`. |

## Navigation is generated — do not hand-edit

Fern needs two navigation files with *different* path conventions, and only one of them is
validated by `fern check`:

| File | Path convention | Read by |
|---|---|---|
| `docs/fern/versions/nightly.yml` | Relative to itself (`../../<page>`) | Every build; validated by `fern check` |
| `docs/index.yml` | Relative to `docs/` (bare paths) | Only `publish-fern-docs.yml` at release time; **not** CI-validated |

Both are generated from the single source of truth, `docs/fern/nav.json`, by
`docs/fern/scripts/gen-nav.mjs`. Editing either output by hand is how a page silently vanishes from
one channel while looking fine in the other.

To add a page:

```bash
# 1. Create the page under docs/, copying the SPDX header block from any existing page.
#    (The header is an MDX comment: {/* ... */} — keep it verbatim as the first lines.)

# 2. Add an entry to docs/fern/nav.json under the right section.

# 3. Regenerate both nav files.
cd docs/fern
npm run nav:gen

# 4. Validate.
npm run check
```

`gen-nav.mjs` also asserts that every page referenced by `nav.json` exists on disk and fails with an
explicit list if not — catch it here rather than in `fern check` or lychee, where the message is far
less obvious. Commit `nav.json`, `versions/nightly.yml`, and `index.yml` together.

## Generating the Python API reference

The `Full-Library-Reference` section under `docs/fern/product-docs/` is generated, gitignored, and
rebuilt on every publish. You do not need it locally for `fern check` to pass — only to see the API
pages in the dev server.

Generating from a local checkout requires a temporary edit to `docs/fern/docs.yml`: uncomment the
`nemo-voice-agent-local` library block.

```yaml
  nemo-voice-agent-local:
    input:
      path: ../../nemo_voice_agent
    output:
      path: ./product-docs/nemo-labs-voice-agent/Full-Library-Reference
    lang: python
```

```bash
cd docs/fern
# 1. Uncomment the nemo-voice-agent-local block in docs.yml.
npm run generate:library:local
# 2. Comment it back out before running check / dev / preview / publish.
```

Re-commenting is mandatory: `fern docs dev` rejects path-backed libraries during config load with
`Library 'nemo-voice-agent-local' uses 'path' input which is not yet supported`. The generated pages
under `product-docs/` survive the re-comment, so the dev server still serves them.

`generate:library:local` runs `npm run sanitize:generated` afterwards. That script rewrites
MDX-invalid JSX attributes the Python generator emits for Pydantic field types — if you ever run the
generator directly, run the sanitizer too.

## Running the dev server

```bash
cd docs/fern
npm run login   # once, or whenever the theme fetch 403s
npm run dev
```

If login fails with an organization access error, sign in at
`https://dashboard.buildwithfern.com` with an account that has NVIDIA org access, then retry.

## Authoring rules CI enforces

Three gates run on any PR touching `docs/` (`fern-docs-ci.yml`):

1. **Every raw HTML `img` tag must be self-closing** anywhere under `docs/`. Prefer plain Markdown
   images; if you must use HTML, close the tag with a trailing slash.
2. **`fern check`** must pass.
3. **lychee `--offline`** over `docs/**/*.md`. Every relative link target must exist on disk. There
   is no ignore file — a link to a repo source file that is not a docs page will fail the build.
   Name source files in inline code instead of linking to them.

Beyond the gates, two authoring conventions matter:

- Write pages as `.md`. The `.mdx` extension is reserved for generated output.
- Fern renders `.md` through MDX, so a bare `{`, `}`, or `<` outside a fenced code block breaks the
  build. Put YAML braces, OmegaConf interpolation, type generics, and comparison operators inside
  fenced code blocks or inline backticks.

A separate pair of workflows (`fern-docs-preview-build.yml` and `fern-docs-preview-comment.yml`)
builds a hosted preview and posts the link as a PR comment. The build job runs on the PR branch with
no secrets; the comment job picks up the artifact and builds with the org token, so previews are safe
on fork PRs.

## Publishing

**Merging a docs change to `main` publishes the nightly channel live.** `publish-fern-docs.yml` fires
on any push to `main` that touches `docs/**`, gated on the repository variable `PUBLISH_FERN=true`.
There is no manual approval step — review the preview before merging.

Releases are separate. On a published GitHub Release (or a manual dispatch with a tag), the same
workflow freezes that tag's docs into a versioned channel, registers it in `docs.yml`, prunes to the
three most recent versions, and opens a PR to persist the registry change back to `main`.
Pre-release tags publish but skip version registration.

## Related

- [Contributing](index.md) — branch, lint, and PR conventions for the repository.
- [Testing](testing.md) — running the pytest suites.
