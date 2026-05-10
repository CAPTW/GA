# Prompt Template: Update README

Task:
Update `README.md` after a change to problems, operators, configs, or experiment workflows.

Checklist:

1. Update the implemented feature list if capabilities changed.
2. Update config reference when names or options changed.
3. Add or remove config examples when the runnable set changed.
4. Mention new scripts or output artifacts in Quickstart or Notes.
5. Keep examples aligned with real file paths in the repository.
6. Run `python -m pytest` after the README-related code changes are complete.

Done criteria:

- README reflects the current operator, problem, and script surface.
- New user-facing config names are documented.
- No stale commands or missing files remain in examples.
