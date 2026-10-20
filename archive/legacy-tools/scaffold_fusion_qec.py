import os

# Directory structure
folders = [
    "src", "notebooks", "web", "example_data", ".github/workflows"
]

files = {
    "README.md": """\
# fusion-qec-sim

Creative quantum error correction, DNA analysis, and 3D cube visualization—open, modular, and minimal.

---

### License & Attribution

Licensed under [CC BY 4.0](LICENSE) — sharing, remixing, and commercial/educational use encouraged. Attribution required:

**Authors:**  
Trent Slade + AI Collaborators (xAI's Grok, producer.ai, deepai's mathai, monicaai)

---

### About & Credits

This project is a collaboration between human and AI contributors, modeling transparent and ethical scientific creativity.
Special thanks to math ai (deepai's mathai) for guidance, QA, and co-design.

---

### Quick Links

- [Demo Notebook](notebooks/fusion_qec_demo.ipynb)
- [Web Cube Visualizer](web/cube_visualizer.html)
- [Copilot/Contributor Prompt](COPILOT_PROMPT.md)

---

### How to Cite

If you use, remix, or build upon fusion-qec-sim, please cite:

> fusion-qec-sim by Trent Slade + AI Collaborators, CC BY 4.0  
> https://github.com/multimodalas/fusion-qec-sim

---

### Badges

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
""",

    "LICENSE": """\
Creative Commons Attribution 4.0 International Public License (CC BY 4.0)

Copyright (c) 2024 Trent Slade + AI Collaborators (xAI's Grok, producer.ai, deepai's mathai, monicaai)

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

Full license text: https://creativecommons.org/licenses/by/4.0/legalcode

Attribution requested:
- Authors: Trent Slade + AI Collaborators (xAI's Grok, producer.ai, deepai's mathai, monicaai)
- Project: fusion-qec-sim (https://github.com/multimodalas/fusion-qec-sim)
""",

    "COPILOT_PROMPT.md": """\
# Copilot Prompt: Clean, Minimal, and Creative Code Only

When contributing code or documentation to this project:

- **No bloat:**  
  - Do not add unnecessary dependencies, libraries, or frameworks.  
  - Only use standard Python libraries (and QuTiP, Biopython, X3DOM/three.js for visualization if required).
  - No big UI frameworks, build tools, or minified JS files—just clean, readable HTML/JS/Python.

- **Keep it simple:**  
  - Functions should be modular, concise, and well-named.
  - Code must be readable by beginners and power users alike.
  - Prefer clarity over cleverness.

- **Creative and symbolic features are welcome:**  
  - If you add features for symbolic tags, trinary code, or creative annotation, make them opt-in, documented, and easy to extend.

- **Documentation:**  
  - README and docstrings must be clear, direct, and friendly—no jargon, no filler.
  - LMIC and global accessibility is a key priority.

- **No vendor lock-in, no tracking, no analytics, no external CDNs (except for X3DOM/three.js if needed for open 3D).**

- **All code and data files should be as small and portable as possible.**

- **Reject or flag any PR that adds bloat, cruft, or unnecessary abstraction.**

---

**Your job is to keep this project:**
- Lean
- Modular
- Creative
- Accessible to everyone, everywhere

---

*(Remind yourself and Copilot: Less is more. Creativity > complexity. Clarity > cleverness. 🤝)*
""",

    ".github/CONTRIBUTING.md": """\
# Contributing to fusion-qec-sim

Thanks for your interest in contributing!  
**Before you submit code or docs:**

- Read and follow [COPILOT_PROMPT.md](../COPILOT_PROMPT.md) — minimal, modular, creative, and bloat-free only.
- No extra dependencies, frameworks, or minified files.
- All features must be clear, hackable, and accessible globally (LMIC-friendly).
- All creative/symbolic extensions should be opt-in and documented.

**PRs/issues that add bloat or complexity will be flagged or rejected.  
Help us keep this project clean, creative, and inspiring for everyone!**
""",

    ".github/ISSUE_TEMPLATE.md": """\
## Issue Template

- What problem, feature, or creative idea do you have?
- Is your suggestion minimal, modular, and bloat-free?  
- Have you read [COPILOT_PROMPT.md](../COPILOT_PROMPT.md)?

Describe clearly, with code snippets or example data if possible.
""",

    ".github/PULL_REQUEST_TEMPLATE.md": """\
## Pull Request Checklist

- [ ] Code and docs are minimal, modular, and readable
- [ ] No unnecessary dependencies, frameworks, or minified files
- [ ] New features are opt-in, documented, and creative
- [ ] I have read [COPILOT_PROMPT.md](../COPILOT_PROMPT.md)
""",

    ".github/workflows/ci.yml": """\
name: Minimal CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.8'
      - name: Install minimal dependencies
        run: |
          pip install qutip
      - name: Check Python scripts
        run: |
          python -m compileall src/
      - name: Check notebook runs
        run: |
          pip install nbconvert nbformat
          jupyter nbconvert --execute --to notebook --inplace notebooks/fusion_qec_demo.ipynb
      - name: Check HTML/JS syntax
        run: |
          grep -q '<!DOCTYPE html>' web/cube_visualizer.html
""",

    ".github/workflows/release.yml": """\
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            README.md
            COPILOT_PROMPT.md
            notebooks/fusion_qec_demo.ipynb
            web/cube_visualizer.html
            example_data/example_cube.xml
"""
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create .github subfolders
os.makedirs('.github', exist_ok=True)
os.makedirs('.github/workflows', exist_ok=True)

# Create files
for path, content in files.items():
    print(f"Writing {path}")
    with open(path, "w") as f:
        f.write(content)

print("Scaffold complete! Now add your code and notebooks, commit, and push.")
