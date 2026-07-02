# NeMo Voice Agent Fern Docs

This directory holds the Fern configuration for the NeMo Voice Agent documentation site.

The docs content lives in `../`, with navigation defined in `../index.yml`. Run Fern commands from this directory so `docs.yml` can resolve paths correctly.

## Quick Links

| What | Where |
|---|---|
| Published site | https://docs.nvidia.com/nemo/voice-agent |
| Fern dashboard | https://dashboard.buildwithfern.com |
| Fern config | `docs.yml` |
| Docs navigation | `../index.yml` |
| CI workflows | `../../.github/workflows/fern-docs-*.yml` |

## Local Setup

Install Node.js 20 or newer, then run these commands from this directory:

```bash
cd docs/fern

npm exec --yes --package "fern-api@$(node -p "require('./fern.config.json').version")" -- fern login
```

If login fails with an organization access error, sign in to https://dashboard.buildwithfern.com with an account that has access to the NVIDIA Fern organization, then run the login command again.

## Validate

Run Fern's config and docs validation:

```bash
npm exec --yes --package "fern-api@$(node -p "require('./fern.config.json').version")" -- fern check
```

## Local Preview

Start the local docs server:

```bash
npm exec --yes --package "fern-api@$(node -p "require('./fern.config.json').version")" -- fern docs dev
```

Fern serves the local preview at http://localhost:3000.

## Hosted Preview

Hosted previews require `DOCS_FERN_TOKEN`:

```bash
export FERN_TOKEN="$DOCS_FERN_TOKEN"
npm exec --yes --package "fern-api@$(node -p "require('./fern.config.json').version")" -- fern generate --docs --preview --id local-preview
```

GitHub Actions also creates preview links when `PUBLISH_FERN_PREVIEWS=true` is set for the repository or organization.
