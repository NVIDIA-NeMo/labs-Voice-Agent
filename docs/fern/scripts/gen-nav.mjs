// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Generate BOTH navigation files from docs/fern/nav.json.
//
//   node scripts/gen-nav.mjs           # write the files
//   node scripts/gen-nav.mjs --check   # exit 1 if they are out of date
//
// Fern needs two navigation files with DIFFERENT path conventions, and only
// one of them is validated by `fern check`:
//
//   versions/nightly.yml  paths relative to itself      -> ../../<page>
//                         read by fern on every build; `fern check` validates it
//
//   ../index.yml          paths relative to docs/       -> <page>
//                         read only by publish-fern-docs.yml at release time,
//                         which rewrites it with:
//                             sed "s|path: |path: ${TAG}-content/|g"
//                         NOT validated by CI
//
// Two consequences the sed imposes on index.yml, both handled here:
//   * exactly ONE space after `path:` (the sed matches the literal "path: ")
//   * `folder:` is NOT rewritten, so it must already be written relative to
//     docs/fern/versions/ — i.e. identical in both files.

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const FERN_DIR = resolve(HERE, "..");
const DOCS_DIR = resolve(FERN_DIR, "..");
const NAV_JSON = join(FERN_DIR, "nav.json");
const NIGHTLY = join(FERN_DIR, "versions", "nightly.yml");
const INDEX = join(DOCS_DIR, "index.yml");

const HEADER = `# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
`;

const banner = (which) =>
  `#
# GENERATED FILE — DO NOT EDIT BY HAND.
# Source: docs/fern/nav.json   Generator: docs/fern/scripts/gen-nav.mjs
# Regenerate: npm --prefix docs/fern run nav:gen
#
# ${which}
#
`;

const nav = JSON.parse(readFileSync(NAV_JSON, "utf8"));

// `prefix` is prepended to every page path; `folder` is never rewritten by the
// release-time sed, so it is emitted identically in both files.
function render(prefix, which) {
  const L = [HEADER, banner(which), "navigation:"];

  if (nav.root) {
    L.push(`  - page: ${nav.root.title}`);
    L.push(`    path: ${prefix}${nav.root.path}`);
    L.push(`    slug: ""`);
  }

  const renderSection = (sec, indent) => {
    const lead = " ".repeat(indent);
    const contentsLead = " ".repeat(indent + 2);
    const entryLead = " ".repeat(indent + 4);

    L.push(`${lead}- section: ${sec.title}`);
    if (sec.slug) L.push(`${contentsLead}slug: ${sec.slug}`);
    L.push(`${contentsLead}contents:`);

    for (const p of sec.pages ?? []) {
      L.push(`${entryLead}- page: ${p.title}`);
      L.push(`${entryLead}  path: ${prefix}${p.path}`);
      if (p.slug) L.push(`${entryLead}  slug: ${p.slug}`);
    }
    for (const sub of sec.subsections ?? []) renderSection(sub, indent + 4);

    if (sec.includeAutodoc) {
      L.push(`${entryLead}# Generated Python API reference. Path is relative to`);
      L.push(`${entryLead}# docs/fern/versions/ in BOTH files — the release sed rewrites`);
      L.push(`${entryLead}# only "path: ", never "folder: ".`);
      L.push(`${entryLead}- folder: ${nav.autodocFolder}`);
    }
  };

  for (const sec of nav.sections) renderSection(sec, 2);
  return L.join("\n") + "\n";
}

// Every referenced page must exist on disk, or lychee and `fern check` will
// fail later with a much less obvious message.
const missing = [];
const seen = new Set();
const check = (p) => {
  if (seen.has(p)) return;
  seen.add(p);
  if (!existsSync(join(DOCS_DIR, p))) missing.push(p);
};
if (nav.root) check(nav.root.path);
const checkSection = (sec) => {
  (sec.pages ?? []).forEach((p) => check(p.path));
  (sec.subsections ?? []).forEach(checkSection);
};
nav.sections.forEach(checkSection);

const outputs = [
  [NIGHTLY, render("../../", "Nightly channel navigation (paths relative to docs/fern/versions/).")],
  [INDEX, render("", "Frozen-release navigation (paths relative to docs/; rewritten by the release sed).")],
];

const isCheck = process.argv.includes("--check");
let stale = 0;
for (const [file, content] of outputs) {
  const cur = existsSync(file) ? readFileSync(file, "utf8") : null;
  if (cur === content) continue;
  stale++;
  if (isCheck) console.error(`out of date: ${file}`);
  else {
    writeFileSync(file, content);
    console.log(`wrote ${file}`);
  }
}

if (missing.length) {
  console.error(`\n${missing.length} page(s) referenced by nav.json but missing on disk:`);
  missing.forEach((p) => console.error(`  docs/${p}`));
  process.exit(1);
}

if (isCheck && stale) {
  console.error("\nRun: npm --prefix docs/fern run nav:gen");
  process.exit(1);
}
console.log(isCheck ? "nav files up to date" : `nav: ${seen.size} pages across ${nav.sections.length} sections`);
