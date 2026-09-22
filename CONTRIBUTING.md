# Contributing to Voice Shield AI

Thank you for your interest in contributing to **Voice Shield AI**! We welcome bug reports, feature suggestions, acoustic physics enhancements, and architectural contributions.

---

## 1. Project Tracking with GitHub Issues

We use [GitHub Issues](https://github.com/aayash317-svg/Ai_voice-_cloning/issues) to track all bugs, feature requests, and sprint tasks:
- **Bug Reports**: Use the `[BUG]` template to describe reproducible acoustic or UI anomalies.
- **Feature Requests**: Use the `[FEAT]` template to suggest new model architectures, telemetry metrics, or integrations.
- **Project Tasks**: Use the `[TASK]` template to assign work items, sprint goals, and milestone deliverables.

---

## 2. Development Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- `ffmpeg` (for universal audio decoding)
- Git

### Setup Steps
1. Fork and clone the repository:
   ```bash
   git clone https://github.com/aayash317-svg/Ai_voice-_cloning.git
   cd Ai_voice-_cloning
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Running & Testing

### Launch Local Development Server
```bash
python app.py
```
Dashboard will be live at `http://127.0.0.1:8000` (or `8050` if 8000 is occupied).

### Run Test Suite
Always ensure all unit tests pass before submitting a Pull Request:
```bash
pytest tests/ -v
```

---

## 4. Pull Request Guidelines

1. **Branch Naming**:
   - `feature/your-feature-name`
   - `fix/issue-description`
   - `docs/documentation-update`
2. **Commit Hygiene**:
   - Write clear, conventional commit messages (`feat: ...`, `fix: ...`, `docs: ...`, `test: ...`).
3. **Documentation**:
   - Update `README.md` and `APPLICATION_DOCUMENTATION.md` if your change introduces new endpoints, acoustic algorithms, or configuration parameters.
