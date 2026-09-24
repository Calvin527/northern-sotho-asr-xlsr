# Manifests

Generated CSV manifests are intentionally excluded from version control by default.

Expected columns:

- `speaker`
- `age`
- `gender`
- `location`
- `filename`
- `audio_path`
- `duration`
- `reference`
- `reference_norm`

The official NCHLT test manifest should contain **2,829 utterances** and **14,133 normalized reference word tokens**.

The report documents **50,655** training and **5,629** validation utterances, but the exact historical membership of the internal split was not archived in the report.
