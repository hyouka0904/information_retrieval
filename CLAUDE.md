## Git Commit Guidelines

### Format
Follow Conventional Commits 1.0.0: `<type>(<scope>): <description>` (scope optional).

Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.

### Message rules
- Write in English, imperative mood, lowercase type and scope.
- Subject ≤ 72 chars, no trailing period.
- Describe the actual change, not the task request. Avoid vague subjects like "update" or "fix bug".
- Add a body when the motivation or details are not obvious. Separate subject, body, and footer with blank lines.
- Mark breaking changes with `!` and a `BREAKING CHANGE:` footer.
- Never fabricate issue references or test results.

### Workflow
- Check `git status` and `git diff` before staging; review staged changes before committing.
- One logical change per commit; split unrelated changes.
- Never commit unrelated user modifications, secrets, credentials, or temp files.
- Run relevant tests before committing. Never bypass Git hooks.
- Never amend, rebase, or force-push without explicit permission.
- Pushing to a remote is allowed directly on working branches (feature/topic branches). Never push directly to main, master, or other default/protected branches without explicit permission.
