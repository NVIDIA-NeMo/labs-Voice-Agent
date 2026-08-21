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

This site uses [Fern](https://buildwithfern.com). Page content lives in `docs/` as Markdown, and the Fern
configuration lives in `docs/fern`. Run the commands on this page from `docs/fern/` or the repository root
with `npm --prefix docs/fern run <script>`.

## Prerequisites

Before you build or preview the documentation, review the requirements for the task you plan to run:

| Requirement | Why |
|---|---|
| Node.js 20 or newer, installed with a version manager | Required for every documentation task. Continuous integration (CI) pins Node 20. Use Node 22.13 or newer for `npm run dev`. |
| Docker | Only for `npm run generate:library:local`, which parses the Python package to build the application programming interface (API) reference. Nothing else needs it. |
| A Fern account with NVIDIA organization access | Only for the development server, hosted previews, and publishing. `docs.yml` sets `global-theme: nvidia`, which the Fern command-line interface (CLI) fetches at serve time. |

The Node.js version affects tasks as follows:

- **CI validation:** `docs/fern/package.json` declares `engines.node >= 20`, and CI pins Node 20 for
  `npm run check`. Node 18 fails with `ReferenceError: crypto is not defined`.
- **Development server:** Use Node 22.13 or newer for `npm run dev`. The server installs the current `pnpm`
  release, which declares `node >= 22.13`.
- **Installation:** Use `nvm` or another version manager. The development server installs `pnpm` globally.
  Installation fails with `EACCES` when npm uses a root-owned global prefix such as `/usr/local`.

The Fern CLI is **not** installed globally. Every npm script uses `npm exec` to fetch and run the exact
version pinned in `docs/fern/fern.config.json`, so local runs match CI.

## Scripts

Use these scripts from `docs/fern` to generate, validate, preview, or publish the site:

| Command | Purpose |
|---|---|
| `npm run nav:gen` | Regenerates both navigation files from `nav.json`. Run after any page add/remove/rename. |
| `npm run nav:check` | Exits non-zero if the generated nav files are stale. |
| `npm run check` | Runs `fern check` to validate `docs.yml` and the nightly navigation. No Docker, login, or generated API reference is required. |
| `npm run login` | Authenticates the CLI so it can fetch the NVIDIA theme. |
| `npm run dev` | Local dev server at `http://localhost:3000`. |
| `npm run generate:library:local` | Builds the Python API reference from the local checkout (Docker). |
| `npm run generate:library` | Builds the API reference from the GitHub source Fern clones. Requires login. |
| `npm run preview` | Hosted preview build. Needs `FERN_TOKEN`. |

## Navigation Is Generated — Do Not Hand-Edit

Fern requires two navigation files with *different* path conventions, and only one of them is
validated by `fern check`:

| File | Path Convention | Consumer |
|---|---|---|
| `docs/fern/versions/nightly.yml` | Relative to itself (`../../<page>`) | Every build. Validated by `fern check`. |
| `docs/index.yml` | Relative to `docs/` (bare paths) | Only `publish-fern-docs.yml` at release time. **Not** CI-validated. |

Both are generated from the single source of truth, `docs/fern/nav.json`, by
`docs/fern/scripts/gen-nav.mjs`. Manual edits can make a page disappear from one channel while it remains
visible in the other.

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

`gen-nav.mjs` also asserts that every page referenced by `nav.json` exists on disk. If a page is missing, the
script lists it explicitly. This message is clearer than the corresponding `fern check` or lychee output.
Commit `nav.json`, `versions/nightly.yml`, and `index.yml` together.

## Generating the Python API Reference

The `Full-Library-Reference` section under `docs/fern/product-docs/` is generated, gitignored, and
rebuilt on every publish. You do not need it locally for `fern check` to pass. Generate it only when you
need to view the API pages in the development server.

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

`generate:library:local` then runs `npm run sanitize:generated`. That script rewrites MDX-invalid JavaScript
XML (JSX) attributes that the Python generator emits for Pydantic field types. If you run the generator directly,
also run the sanitizer.

## Running the Dev Server

To authenticate Fern and start the local documentation server, run:

```bash
cd docs/fern
npm run login   # once, or whenever the theme fetch 403s
npm run dev
```

If login fails with an organization access error, sign in to the
[Fern dashboard](https://dashboard.buildwithfern.com) with an account that has NVIDIA organization access,
and then retry.

## Authoring Rules CI Enforces

Three gates run on any pull request (PR) that touches `docs/` (`fern-docs-ci.yml`):

1. **Every raw HTML `img` tag must be self-closing** anywhere under `docs/`. Prefer plain Markdown
   images. If you must use HTML, close the tag with a trailing slash.
2. **`fern check`** must pass.
3. **lychee `--offline`** over `docs/**/*.md`. Every relative link target must exist on disk. There
   is no ignore file—a link to a repository source file that is not a documentation page fails the build.
   Name source files in inline code instead of linking to them.

Beyond the gates, two authoring conventions matter:

- Write pages as `.md`. The `.mdx` extension is reserved for generated output.
- Fern renders `.md` through MDX, so a bare `{`, `}`, or `<` outside a fenced code block breaks the
  build. Put YAML braces, OmegaConf interpolation, type generics, and comparison operators inside
  fenced code blocks or inline backticks.

A separate pair of workflows (`fern-docs-preview-build.yml` and `fern-docs-preview-comment.yml`)
builds a hosted preview and posts the link as a PR comment. The build job runs on the PR branch with
no secrets. The comment job picks up the artifact and builds with the organization token, so previews are
safe on fork pull requests.

## Publishing

**Merging a documentation change to `main` publishes the nightly channel live.** `publish-fern-docs.yml` fires
on any push to `main` that touches `docs/**`, gated on the repository variable `PUBLISH_FERN=true`.
There is no manual approval step—review the preview before merging.

Releases are separate. For a published GitHub Release or a manual dispatch with a tag, the same
workflow freezes that tag's documentation into a versioned channel, registers it in `docs.yml`, prunes to the
three most recent versions, and opens a pull request to persist the registry change back to `main`.
Pre-release tags publish but skip version registration.

## Related

Use these pages for the repository contribution workflow and test suites:

- [Contributing](index.md): Branch, lint, and pull request conventions for the repository.
- [Testing](testing.md): How to run the pytest suites.
