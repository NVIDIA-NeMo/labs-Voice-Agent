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

import { readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, relative, sep } from "node:path";

const root =
  process.argv[2] ??
  "product-docs/nemo-labs-voice-agent/Full-Library-Reference";

const escapeTemplate = (value) =>
  value
    .replaceAll("\\", "\\\\")
    .replaceAll("`", "\\`")
    .replaceAll("${", "\\${");

async function* mdxFiles(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      yield* mdxFiles(path);
    } else if (entry.isFile() && path.endsWith(".mdx")) {
      yield path;
    }
  }
}

let changed = 0;
const files = [];
const routes = new Map();

for await (const file of mdxFiles(root)) {
  files.push(file);
  const content = await readFile(file, "utf8");
  const slug = content.match(/^slug:\s*(.+)$/m)?.[1];
  if (slug != null) {
    routes.set(`/${slug}`, file);
  }
}

for (const file of files) {
  const before = await readFile(file, "utf8");
  const sanitized = before
    .split("\n")
    .map((line) => {
      const sanitizedLine = line
        .replace(/``([^`\n]+)``/g, "`$1`")
        .replace(/:[A-Za-z]+:`([^`\n]+)`/g, "`$1`");

      if (
        !sanitizedLine.includes("<ParamField") ||
        !sanitizedLine.includes(' type="')
      ) {
        return sanitizedLine;
      }

      const match = sanitizedLine.match(/^(.*\stype=")(.*)(">\s*)$/);
      if (match == null) {
        return sanitizedLine;
      }

      const left = match[1].slice(0, -1);
      const value = escapeTemplate(match[2]);
      return `${left}{\`${value}\`}>`;
    })
    .join("\n");
  const after = sanitized.replace(
    /\]\((\/[^\s)#?]+)(#[^)]+)?\)/g,
    (link, route, fragment = "") => {
      const target = routes.get(route);
      if (target == null) {
        return link;
      }

      let targetPath = relative(dirname(file), target).split(sep).join("/");
      if (!targetPath.startsWith(".")) {
        targetPath = `./${targetPath}`;
      }
      return `](${targetPath}${fragment})`;
    },
  );

  if (after !== before) {
    await writeFile(file, after);
    changed += 1;
  }
}

console.log(`Sanitized generated MDX in ${changed} file(s).`);
