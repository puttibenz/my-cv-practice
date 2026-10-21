# Project Rules & Workflow (AGENTS.md)

## 1. Automated Progress Tracking & Git Sync
Whenever the user completes a module and passes code review:
1. Mark the module as `[x]` completed in `STUDY_PLAN.md`.
2. Run `git add`, `git commit`, and `git push origin main` immediately so GitHub is always up to date.

## 2. Exercise Style & Format
- Use `.py` files with `# %%` cell markers for interactive execution in VS Code / Antigravity.
- Every exercise must include:
  - Clear Thai docstrings and mathematical explanations.
  - Type hints.
  - Automated `assert` tests verifying array shapes, data types (`uint8` vs `float32`), and numerical accuracy.
  - Optional/Interactive `matplotlib` visualization helper so the user can visually inspect Before/After images.

## 3. Zero-Friction Datasets
- Never require manual dataset downloads from external websites.
- Always generate test images programmatically or auto-download built-in datasets (`torchvision`, `ultralytics`, `skimage` / `opencv` samples) via Python code.
