# Released Results

This directory contains compact, paper-ready outputs from the frozen Scientific Reports experiment suite.

- `paper_ready/`: dataset-level summaries, ranks, ablations, sensitivity analyses, statistical tests, runtimes, and the final audit report.
- `supplementary_tables/`: manuscript-ready supplementary tables.
- `figure_source_tables/`: compact source tables used by the confirmed figure layouts.
- `dataset_manifest.csv`: dataset dimensions and class distributions observed by the frozen loader.

Excluded by design:

- raw public dataset files;
- `all_predictions.csv` and teacher-state sample dumps;
- `.partial.csv` resume checkpoints;
- model checkpoints, logs, caches, temporary archives, Origin projects, PPT conversion files, and duplicate rendered figures.

The excluded files are either large reproducible intermediates, local tooling artifacts, or outside the minimal paper-ready release surface.

