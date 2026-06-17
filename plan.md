# OmniVoice Work Plan & Status

This document tracks the current working status and tasks for development, debugging, and running **OmniVoice**.

---

## Current Status

We are debugging the startup of the Gradio web interface (`omnivoice/cli/demo.py`).

### 🔴 Encountered Issue: `ImportError` on Startup
Running `python .\omnivoice\cli\demo.py` fails with:
```
ImportError: cannot import name 'get_best_device' from 'omnivoice.utils.common' (D:\AI_DATA\OmniVoice\Lib\site-packages\omnivoice\utils\common.py)
```

**Root Cause:**
- Python is executing `demo.py` as a script, which appends `omnivoice/cli/` to the python search path (`sys.path`). It does not add the workspace root `OmniVoice_src/` containing the local `omnivoice` package.
- Python falls back to importing `omnivoice` from the virtual environment's `site-packages` (`D:\AI_DATA\OmniVoice\Lib\site-packages\omnivoice`), which contains an older/incompatible version lacking `get_best_device`.

---

## Proposed Action Plan

### 1. Fix Import Resolution (Immediate)
We have three approaches to resolve the path resolution:

- **Approach A (Recommended): Run as a module** [VERIFIED]
  Run the demo using python's module execution mode from `OmniVoice_src/`:
  ```powershell
  python -m omnivoice.cli.demo --port 8001
  ```
  *(Note: Run with the virtual environment's python path if needed: `D:\AI_DATA\OmniVoice\Scripts\python.exe -m omnivoice.cli.demo --port 8001`)*

- **Approach B: Set `PYTHONPATH`**
  Prepend the current working directory to the python path:
  ```powershell
  $env:PYTHONPATH="."
  python .\omnivoice\cli\demo.py --port 8001
  ```

- **Approach C: Reinstall in Editable Mode**
  Install the local source code into the virtual environment as an editable package:
  ```powershell
  pip install -e .
  # or using uv:
  uv pip install -e .
  ```

### 2. Verify Gradio Server Launch
Once import paths are resolved:
1. Start the Gradio server on target port `8001`.
2. Inspect if required model weights (e.g., `k2-fsa/OmniVoice`) are auto-downloaded or require local paths.
3. Test voice cloning and voice design UI functionalities.

### 3. Implement Needed Changes
- Document the correct running command in `README.md` or a quick-start guide.
- Fix any potential script-level issues if new errors arise during runtime.
