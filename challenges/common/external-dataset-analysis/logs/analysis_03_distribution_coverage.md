# Analysis 03 — Embedding-Based Distribution Coverage Log

**Analyst:** Claude Sonnet 4.6 Agent  
**Date started:** 2026-03-11  
**Notebook:** notebooks/03_distribution_coverage.ipynb

## Task
RADIO-H CLS-token embeddings for all external images → t-SNE on combined LaRS+external → OOD coverage metric.

## Environment note
- lars-analysis conda env no longer exists; using macvi env with packages installed
- cuml GPU t-SNE not available → using openTSNE (fast CPU) as substitute
- RADIO-H model not cached locally → will download ~2-3 GB from HuggingFace on first run

## Status
- [ ] Model downloaded
- [ ] MULTIAQUA embeddings extracted
- [ ] OrcaTech embeddings extracted
- [ ] Tampere embeddings extracted
- [ ] WaterScenes embeddings extracted (subsample 10k for t-SNE only)
- [ ] ROSEBUD embeddings extracted
- [ ] external_embeddings.db saved to shared_data/
- [ ] t-SNE computed on LaRS + all external
- [ ] OOD coverage table (3 thresholds × 6 LaRS sources)

## Decisions / Issues
- **cuml unavailable**: Using openTSNE instead of cuml TSNE. openTSNE uses a fast approximation
  (FFT-accelerated Barnes-Hut) and is significantly faster than sklearn; runtime ~10-30 min for 60k points.
- **WaterScenes subsampling**: use every Nth image for t-SNE only; all images used for coverage metric.
