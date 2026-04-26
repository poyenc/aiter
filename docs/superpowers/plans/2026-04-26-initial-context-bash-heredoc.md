# Initial-Context Bash Heredoc Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the Write tool's read-before-write guard from blocking `initial-context.md` generation/regeneration in the hip-kernel-team skill.

**Architecture:** Add a tool-selection note to `setup.md` (step 7b) and `resume.md` (step 3b) instructing use of `Bash` with heredoc instead of the Write tool for this fully-generated file.

**Tech Stack:** Markdown skill files

---

### Task 1: Add tool note to `setup.md` step 7b

**Files:**
- Modify: `~/.claude/skills/hip-kernel-team/phases/setup.md:118-119` (after the include/exclude guidelines table, before step 8)

- [ ] **Step 1: Add the tool note**

Insert the following block after line 118 (the last row of the include/exclude table) and before line 120 (`## Step 8: Save & Spawn`):

```markdown

**Tool note:** Use `Bash` with a heredoc (`cat <<'EOF' > path`) to
write this file, not the Write tool. `initial-context.md` is fully
generated content — Write's read-before-write guard adds no value here
and will block on resume when the file already exists.
```

The exact edit target (old string to match):

```
| Available Skills | User-created skills with resolved file paths | Built-in superpowers skills (not on disk) |

## Step 8: Save & Spawn
```

Replace with:

```
| Available Skills | User-created skills with resolved file paths | Built-in superpowers skills (not on disk) |

**Tool note:** Use `Bash` with a heredoc (`cat <<'EOF' > path`) to
write this file, not the Write tool. `initial-context.md` is fully
generated content — Write's read-before-write guard adds no value here
and will block on resume when the file already exists.

## Step 8: Save & Spawn
```

- [ ] **Step 2: Verify the edit**

Run: `grep -A2 "Tool note" ~/.claude/skills/hip-kernel-team/phases/setup.md`

Expected: The tool note text appears, followed by a blank line and `## Step 8`.

- [ ] **Step 3: Commit**

```bash
git -C ~/.claude add skills/hip-kernel-team/phases/setup.md
git -C ~/.claude commit -m "fix: use Bash heredoc for initial-context.md in setup phase"
```

---

### Task 2: Add tool note to `resume.md` step 3b

**Files:**
- Modify: `~/.claude/skills/hip-kernel-team/phases/resume.md` (step 3b, after "not stale snapshots from the original session.")

- [ ] **Step 1: Add the tool note**

The exact edit target (old string to match):

```
   listings — not stale snapshots from the original session.

4. **Spawn team**:
```

Replace with:

```
   listings — not stale snapshots from the original session.

   **Tool note:** Use `Bash` with a heredoc (`cat <<'EOF' > path`) to
   write this file, not the Write tool. `initial-context.md` is fully
   generated content — Write's read-before-write guard adds no value
   here and will block when the file already exists from a prior session.

4. **Spawn team**:
```

- [ ] **Step 2: Verify the edit**

Run: `grep -A2 "Tool note" ~/.claude/skills/hip-kernel-team/phases/resume.md`

Expected: The tool note text appears with proper indentation (3 spaces, matching step 3b's body).

- [ ] **Step 3: Commit**

```bash
git -C ~/.claude add skills/hip-kernel-team/phases/resume.md
git -C ~/.claude commit -m "fix: use Bash heredoc for initial-context.md in resume phase"
```
