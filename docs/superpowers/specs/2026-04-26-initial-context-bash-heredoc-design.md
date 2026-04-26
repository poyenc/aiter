# Fix: Use Bash heredoc for initial-context.md writes

**Date:** 2026-04-26
**Scope:** `hip-kernel-team` skill — `phases/setup.md`, `phases/resume.md`

## Problem

The Write tool requires reading a file before overwriting it. `initial-context.md` is fully generated content with no user edits to preserve. On team resume (step 3b), the file already exists from the prior session, causing Write to reject the overwrite. On setup (step 7b), the same issue occurs if a partial failure or re-run leaves the file behind.

## Solution

Add a tool-selection note to both `setup.md` (step 7b) and `resume.md` (step 3b) instructing the lead to use `Bash` with a heredoc instead of the Write tool for `initial-context.md`.

## Changes

### `phases/setup.md` — step 7b

Append after the include/exclude guidelines table:

> **Tool note:** Use `Bash` with a heredoc (`cat <<'EOF' > path`) to write this file, not the Write tool. `initial-context.md` is fully generated content — Write's read-before-write guard adds no value here and will block on resume when the file already exists.

### `phases/resume.md` — step 3b

Append the same note after "Same format as setup step 7b.":

> **Tool note:** Use `Bash` with a heredoc (`cat <<'EOF' > path`) to write this file, not the Write tool. `initial-context.md` is fully generated content — Write's read-before-write guard adds no value here and will block on resume when the file already exists.

## Why not other approaches?

- **Read-then-Write guard:** Wastes a step reading content we're about to discard entirely.
- **Pre-delete then Write:** Fragile — relies on Write tool's internal new-vs-existing file detection.

## Files modified

1. `~/.claude/skills/hip-kernel-team/phases/setup.md`
2. `~/.claude/skills/hip-kernel-team/phases/resume.md`
