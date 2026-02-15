<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

---
name: tvm-ffi-code-review
description: Run parallel code reviews using Claude Code and OpenAI Codex reviewers. Produces a unified, prioritized review report with actionable findings from multiple AI models.
# disable-model-invocation avoids a slow planning turn before the skill starts executing.
# The skill prompt is fully self-contained — every decision tree, diff command, and output
# format is spelled out below — so an extra model invocation only adds latency with no benefit.
disable-model-invocation: true
argument-hint: "[pr (default) | branch:<name> | commit:<sha> | staged | unstaged]"
allowed-tools: Bash(git *), Bash(gh *), Bash(codex *), Bash(command -v *), Bash(jq *), Read, Grep, Glob, Task, AskUserQuestion
---

# Multi-Model Code Review

Review code changes using two independent AI reviewers in parallel — **Claude Code** and **OpenAI Codex** — then synthesize their findings into a single prioritized report.

## Prerequisites

- **Codex CLI** must be installed and authenticated (`npm install -g @openai/codex` or equivalent). If unavailable, the skill gracefully falls back to Claude-only review.
- **jq** must be installed for GitHub PR comment publishing (Step 5). Available via `brew install jq`, `apt install jq`, etc.

## Step 1: Determine review scope

Parse `$ARGUMENTS` to determine what to review. If the argument is empty or ambiguous, prompt the user with `AskUserQuestion` to choose a scope.

### Supported scopes

| Argument | Diff command | Description |
|----------|-------------|-------------|
| `pr` (default) | `git diff <main-branch>...HEAD` | All changes in the current PR/branch since it diverged from the main branch. Auto-detects the main branch (see detection order below). |
| `branch:<name>` | `git diff "<name>"...HEAD` | Changes relative to the named branch (merge-base mode). |
| `commit:<sha>` | `git diff "<sha>"..HEAD` | Changes since the given commit (linear history). |
| `staged` | `git diff --cached` | Only staged (indexed) changes. |
| `unstaged` | `git diff` | Only unstaged working-tree changes. |

> **Note on `...` vs `..`:** The three-dot form (`A...B`) diffs from the merge base of A and B to B — this shows only changes introduced on the current branch, excluding upstream changes. The two-dot form (`A..B`) diffs the two endpoints directly — appropriate for `commit:<sha>` where you want all changes since that exact commit.

### Main branch detection order

For the `pr` scope, detect the main branch by checking (in order):

1. `gh repo view --json defaultBranchRef -q .defaultBranchRef.name` (authoritative, requires network)
2. `git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null` (strip `refs/remotes/origin/` prefix)
3. Check if `main` or `master` exists locally (`git rev-parse --verify main` / `master`)
4. If none found, ask the user via `AskUserQuestion`. If the user does not provide a valid branch, halt with: "Unable to determine main branch. Please specify manually using `branch:<name>` syntax."

### Scope resolution logic

1. If `$ARGUMENTS` is empty, default to `pr` scope.
2. If `$ARGUMENTS` matches one of the keywords above (`pr`, `staged`, `unstaged`, or the `branch:` / `commit:` prefixes), use that scope.
3. If `$ARGUMENTS` doesn't match a keyword, verify it is a valid git ref with `git rev-parse --verify "$ARGUMENTS" 2>/dev/null`. If valid, treat as `branch:<arg>` for backward compatibility. If invalid, fall through to rule 4.
4. If the argument is ambiguous or the ref doesn't exist, ask the user:

<!-- Pseudocode — use the AskUserQuestion tool with these options: -->
```
AskUserQuestion:
  question: "What would you like to review?"
  options:
    - "Current PR (all commits since diverging from main)"
    - "Against a specific branch"
    - "Since a specific commit"
    - "Staged changes only"
    - "Unstaged changes only"
```

### Gather the diff

Once the scope is resolved, run:

```bash
# Always quote user-supplied values to prevent shell injection.
git diff --no-color --stat "$RESOLVED_DIFF_ARGS"
git diff --no-color --unified=5 "$RESOLVED_DIFF_ARGS"
```

Use `--unified=5` (5 lines of context) for standard reviews. This provides enough surrounding code for reviewers to understand intent without bloating large diffs.

If the diff is empty (no changes detected for the given scope), inform the user and stop. Do not launch reviewers on an empty diff.

Store the diff output and the list of changed files. If the diff exceeds ~3000 lines (chosen to fit comfortably within reviewer context windows while covering the majority of typical PRs), truncate it:

1. Parse `git diff --no-color --stat "$RESOLVED_DIFF_ARGS"` to extract per-file change counts (insertions + deletions).
2. Sort files by: total changes descending, then by type priority:
   - **Source code** (priority 0): `*.cc`, `*.cpp`, `*.h`, `*.hpp`, `*.py`, `*.pyx`, `*.pxd`, `*.rs`, `*.go`, `*.js`, `*.ts`
   - **Tests** (priority 1): files under `tests/`, `test/`, or matching `*_test.*`, `test_*.*`
   - **Docs** (priority 2): `*.md`, `*.rst`, `*.txt`, files under `docs/`
3. Accumulate files in sorted order, generating per-file diffs via `git diff --no-color --unified=5 "$RESOLVED_DIFF_ARGS" -- "<file1>" "<file2>" ...`, until reaching ~3000 lines. Each file is included in full or not at all.
4. Include a note in the reviewer prompts: "Large diff detected. Review covers N of M files (X% of total changes). Omitted files: ..."

## Step 2: Launch reviewers

### Pre-flight check for Codex CLI

Before launching reviewers, check if the Codex CLI is available:

```bash
command -v codex &>/dev/null
```

If the command fails, set `CODEX_AVAILABLE=false` and log: "Codex CLI not found. Proceeding with Claude Code review only." Skip the Codex reviewer and launch only the Claude reviewer.

### Claude Code Reviewer

Launch via the **Task** tool. These are Claude Code Task tool parameters — `subagent_type` selects the specialized agent, `model` selects the backing model, and `prompt` provides the review payload:

```
Task(
  description: "Claude review of PR"
  subagent_type: "claude-code-reviewer"
  model: "opus"
  prompt: "<the full unified diff, the list of changed files, and the shared review instruction below>"
)
```

### Codex Code Reviewer (if available)

Launch via the **Task** tool. The `codex-code-reviewer` subagent invokes the Codex CLI via its Bash tool internally:

```
Task(
  description: "Codex review of PR"
  subagent_type: "codex-code-reviewer"
  prompt: "<the full unified diff, the list of changed files, and the shared review instruction below>"
)
```

### Parallel execution

If both reviewers are available, launch **both** Task calls simultaneously in a single response. Each reviewer should complete within a reasonable time. If one reviewer fails or times out, continue with results from the other.

### Handling task failures

- If the Claude task fails: Capture the task's returned error message. Report to the user: "Claude Code reviewer failed: <error message>". Stop.
- If the Codex task fails (or Codex CLI was unavailable): Log "Codex reviewer failed: <error message>". Continue with Claude results only. Note in the Codex section: "Codex review unavailable — proceeding with Claude Code review only."
- If both fail: Report both error messages and stop.

### Shared review instruction

Both reviewers receive the **same** instruction so their findings are directly comparable:

> Review this diff thoroughly. For each finding, provide: severity (critical/high/medium/low/nit), file path, line number, category, description, and a suggested fix or code snippet.
>
> Cover all of the following areas:
> - **Correctness**: Logic errors, off-by-one mistakes, wrong return values, missing edge cases, race conditions
> - **Security**: Injection vulnerabilities, buffer overflows, unsafe deserialization, improper input validation, credential exposure
> - **Performance**: Unnecessary allocations, O(n^2) where O(n) is possible, redundant I/O, missing caching opportunities
> - **API design**: Confusing interfaces, breaking changes, poor naming, missing or misleading documentation
> - **Maintainability**: Dead code, excessive complexity, poor separation of concerns, missing abstractions or premature abstractions
> - **Concurrency**: Data races, deadlocks, unsafe shared state, missing synchronization
> - **Error handling**: Swallowed exceptions, missing error propagation, unclear failure modes, resource leaks
> - **Best practices**: Violations of language idioms, style inconsistencies with the surrounding codebase, deprecated API usage

## Step 3: Present individual reviewer results

After both reviewers return, print each reviewer's full response **verbatim** under its own heading before any synthesis. This lets the user see the raw output from each model.

Format:

```markdown
---

## Claude Code Review

<full response from the claude-code-reviewer subagent, verbatim>

---

## Codex Code Review

<full response from the codex-code-reviewer subagent, verbatim>

---
```

If a reviewer failed or was unavailable, print a note in its section explaining why (e.g., "Codex CLI not found — skipped.").

## Step 4: Synthesize into a unified report

After presenting individual results, merge their findings into one combined report.

**Synthesis rules:**
1. **Deduplicate**: If both reviewers flag the same issue (same file, similar line range, same category), merge into a single "consensus" finding — these get elevated confidence.
2. **Sort by severity**: critical > high > medium > low > nit.
3. **Preserve provenance**: Tag each finding with its source (Claude, Codex, or Consensus).
4. **Keep actionable details**: Preserve suggested fixes, code snippets, and unified diff patches.
5. **Note divergences**: If reviewers disagree on severity or approach, present both perspectives.
6. **Graceful degradation**: If one reviewer failed (e.g., Codex CLI not installed), note it and present results from the available reviewer only.

### Output format

```markdown
---

## Synthesized Code Review Report

**Scope**: `<scope description>` | **Files changed**: N | **Reviewers**: Claude Code, Codex

### Consensus Findings
Issues flagged by both reviewers (high confidence):
- **file:line** — description (severity) — suggested fix

### Critical / High
| # | File:Line | Category | Finding | Source | Suggested Fix |
|---|-----------|----------|---------|--------|---------------|

### Medium
| # | File:Line | Category | Finding | Source | Suggested Fix |
|---|-----------|----------|---------|--------|---------------|

### Low / Nits
- ...

### Reviewer Divergences
Cases where reviewers disagree (if any) — present both perspectives.
```

## Step 5: Offer to publish review to GitHub

After presenting the synthesized report, if the review scope is `pr` and the current branch has an open pull request, offer to publish the review as inline GitHub PR review comments.

Ask the user:

<!-- Pseudocode — use the AskUserQuestion tool with these options: -->
```
AskUserQuestion:
  question: "Would you like to publish this review as inline comments on the GitHub PR?"
  options:
    - "Yes — post as inline review comments"
    - "No — keep local only"
```

If the user declines, stop here. Otherwise, first verify `jq` is available (`command -v jq &>/dev/null`). If not, inform the user: "`jq` is required for GitHub publishing but was not found. Install it and retry." Then submit a pull request review with inline comments placed on the relevant diff lines:

1. **Detect the PR number and repo**: Run `gh pr view --json number,headRefOid` and `gh repo view --json nameWithOwner -q .nameWithOwner` to get the PR number, head SHA, and `$OWNER/$REPO`. Validate that the PR number is numeric and the head SHA is at least 7 hex characters.
2. **Verify local HEAD matches remote**: Compare `headRefOid` from step 1 with `git rev-parse HEAD`. If they differ, warn the user: "Local HEAD does not match the PR head on GitHub. Push your changes first, or the review comments may be placed incorrectly." Ask the user whether to continue or abort.
3. **Map findings to diff positions**: Fetch the PR diff once and store it: `PR_DIFF="$(gh pr diff "$PR_NUMBER")"`. For each finding with a specific file and line number, verify that line appears in the cached diff output. Only lines that are part of the diff can receive inline comments.
4. **Build and submit the review payload**: Use `jq` for safe JSON construction to avoid injection from reviewer-generated content, and pipe directly to `gh api` via stdin. **IMPORTANT**: Do NOT use `--field 'comments=[...]'` — `gh api --field` treats array values as strings, causing a 422 error. Always use `--input -` with stdin piping instead.

#### Payload construction and submission

Use `jq` to safely construct the JSON payload and pipe it directly to `gh api` — no temporary files needed:

```bash
jq -n \
  --arg commit_id "$HEAD_SHA" \
  --arg body "$REVIEW_SUMMARY" \
  --argjson comments "$COMMENTS_JSON" \
  '{commit_id: $commit_id, event: "COMMENT", body: $body, comments: $comments}' \
| gh api "repos/$OWNER/$REPO/pulls/$PR_NUMBER/reviews" \
  --method POST \
  --input -
```

Where `$COMMENTS_JSON` is a JSON array built with `jq` for each finding:

```bash
jq -n \
  --arg path "src/example.cc" \
  --arg body "**[high | security]** Description.\n\nSuggested fix: ...\n\n_Source: Consensus_" \
  '{path: $path, line: 42, side: "RIGHT", body: $body}'
```

This ensures all string fields are properly escaped, prevents JSON injection, and avoids temporary file management and cleanup concerns.

**Rules for inline comments:**
- Only post comments on lines that exist in the PR diff (the API will reject others).
- Always include `"side": "RIGHT"` to place comments on the new version of the file.
- Combine multiple findings on the same line into a single comment.
- Each comment body should include: severity badge, category, description, suggested fix, and source attribution.
- The review body (top-level comment) should contain a summary with finding counts by severity and a note that it was generated by multi-model review.
- If a finding cannot be mapped to a diff line (e.g., a general architectural concern), include it in the top-level review body instead.
