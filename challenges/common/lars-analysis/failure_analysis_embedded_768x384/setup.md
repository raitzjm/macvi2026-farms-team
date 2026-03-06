# Setup Log — Phase 1 Failure Analysis

Environment: `macvi` conda env (Python 3.11, macvi-farms:latest container)

## New packages installed (2026-03-03)

The base `macvi` env did not have pandas, plotly, or weasyprint. The following were
installed and tested before use:

```bash
# Python packages
pip install pandas plotly weasyprint kaleido

# System libraries for weasyprint (requires pango/cairo)
sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libpangocairo-1.0-0 \
    libcairo2 libgdk-pixbuf2.0-0 shared-mime-info
```

Versions confirmed working:
- pandas 3.0.1
- plotly 6.6.0
- weasyprint 68.1
- kaleido 1.2.0

## Notes

- kaleido 1.2.0 requires Google Chrome for plotly PNG export → not usable in this
  container. Figures are generated with matplotlib instead.
- Notebook uses Plotly for inline interactive rendering (no PNG export needed).
- matplotlib 3.10.8 was already available in the macvi env.

## Dockerfile update needed

Add to Dockerfile (macvi-farms image):
```dockerfile
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 libpangoft2-1.0-0 libpangocairo-1.0-0 \
    libcairo2 libgdk-pixbuf2.0-0 shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pandas plotly weasyprint kaleido
```
