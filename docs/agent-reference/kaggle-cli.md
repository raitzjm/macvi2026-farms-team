# Kaggle CLI & Python SDK — Agent Reference

## Installation & Auth

The `kaggle` package (v2.1.0+) is installed globally in the container and includes both the CLI and `kagglesdk` Python bindings. No separate install needed.

Auth is provided via the `KAGGLE_API_TOKEN` environment variable (set by `run.sh` from the host). No `kaggle.json` file required.

```bash
kaggle --version          # verify CLI works
kaggle competitions list  # verify auth works
```

---

## CLI — Competition Commands

### Fetch competition page content (rules, overview, data, etc.)
```bash
# List all available pages for a competition
kaggle competitions pages fathomnet-2026

# Fetch a specific page verbatim (returns markdown)
kaggle competitions pages fathomnet-2026 --content --page-name rules
kaggle competitions pages fathomnet-2026 --content --page-name Guidelines
kaggle competitions pages fathomnet-2026 --content --page-name Evaluation
kaggle competitions pages fathomnet-2026 --content --page-name data-description
kaggle competitions pages fathomnet-2026 --content --page-name Timeline

# Available pages (fathomnet-2026):
# data-description, Description, Evaluation, rules, abstract,
# foundational-rules, "CVPR & CLEF Content 2026", Acknowledgements,
# Guidelines, Timeline
```

### Submissions
```bash
# Submit a CSV file
kaggle competitions submit -c fathomnet-2026 -f submission.csv -m "description"

# View your past submissions
kaggle competitions submissions -c fathomnet-2026
```

### Leaderboard
```bash
# Show current leaderboard
kaggle competitions leaderboard -c fathomnet-2026 --show
```

### Files
```bash
# List competition files
kaggle competitions files -c fathomnet-2026

# Download a specific file
kaggle competitions download -c fathomnet-2026 -f dataset_train.json
```

### Kernels (notebook training runs)
```bash
# Initialize kernel metadata in a directory
kaggle kernels init -p /path/to/kernel/

# Push and trigger a kernel run
kaggle kernels push -p /path/to/kernel/

# Check kernel run status
kaggle kernels status <username>/<kernel-slug>

# Pull kernel output files
kaggle kernels output <username>/<kernel-slug>

# Fetch kernel console logs (historical or live follow mode)
kaggle kernels logs <username>/<kernel-slug>
kaggle kernels logs <username>/<kernel-slug> --follow
```

#### kernel-metadata.json required fields
```json
{
  "id": "username/kernel-slug",
  "title": "My Training Run",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "accelerator": "NvidiaTeslaT4",
  "dataset_sources": ["competitions/fathomnet-2026"]
}
```

**Available accelerators:** `NvidiaTeslaP100` (default), `NvidiaTeslaT4`, `NvidiaTeslaT4Highmem`, `NvidiaTeslaA100`

---

## Python SDK (`kagglesdk`)

`kagglesdk` is bundled with the `kaggle` package — no separate install. Auth uses the same `KAGGLE_API_TOKEN` env var.

### Fetch competition metadata
```python
import kaggle

api = kaggle.KaggleApiExtended()
api.authenticate()

comp = api.competition_get('fathomnet-2026')
print(comp.title, comp.deadline, comp.max_daily_submissions)
```

### Fetch competition page content
```python
pages = api.competition_list_pages('fathomnet-2026')
for page in pages:
    print(page.name)
    print(page.content)  # full verbatim markdown text
```

### Fetch discussion threads
```python
# List all discussion topics
from kagglesdk.competitions.services.competition_api_service import CompetitionApiService
from kagglesdk.kaggle_api_extended import KaggleApiExtended

api = KaggleApiExtended()
api.authenticate()

# List topics (returns topic list with id, title, vote count, comment count)
comp_client = api.competitions_client
topics = comp_client.list_topics('fathomnet-2026')

# Fetch messages in a specific topic (e.g. discussion #688810)
messages = comp_client.list_topic_messages('fathomnet-2026', 688810)
for msg in messages:
    print(msg.posted_by_ref.user_name, msg.content)
```

### Submit predictions
```python
api.competition_submit(
    file_name='submission.csv',
    message='Run description',
    competition='fathomnet-2026'
)
```

---

## Key Facts (fathomnet-2026)

| Field | Value |
|---|---|
| Competition ID | 125799 |
| Max team size | 10 |
| Max daily submissions | 10 |
| Final submissions selected | 2 |
| Evaluation metric | mAP@[0.50:0.95] |
| Deadline | 2026-05-07 23:59 CET |

---

## Notes for Agents

- Discussion threads are NOT accessible via the CLI subcommands — use the Python SDK (`list_topic_messages`) or the Kaggle REST API directly
- Competition page content (rules, guidelines, etc.) is best fetched via `kaggle competitions pages --content` — more reliable than web scraping since Kaggle pages are JS-rendered SPAs
- Playwright can scrape the rules page but may fail on other tabs due to React hydration issues in headless mode
- The `KAGGLE_API_TOKEN` env var is set automatically inside the container — no manual auth setup needed
