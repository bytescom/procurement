# Cantill.ai Procurement Data & ML Assessment

This repository contains a configuration-driven procurement pipeline for the supplied Company A and Company B data. It ingests configured CSVs, preserves source rows, normalizes amounts and dates, investigates cross-system matches and duplicates, applies explicit spend rules, resolves supplier/item keys, classifies against the configured taxonomy, and writes the assessment output schema.

## Run

From the project root, install dependencies and run both companies:

```powershell
python -m pip install -r requirements.txt
python -m src.pipeline --all --output-dir output --name candidate
```

Run one company or another config with:

```powershell
python -m src.pipeline --config configs/company_A.yaml --output-dir output --name candidate
```

The pipeline resolves data paths from the project root discovered from `src/ingestion/loader.py`, so it can also be launched from another working directory with an absolute config path. Outputs are named `candidate__A.csv` and `candidate__B.csv`; audit tables are written to `output/audit/{company_id}/`.

Each output uses the exact 11-column assessment schema in `output_template.csv`. The audit directory contains spend decisions, entity-resolution maps, row counts, preprocessing changes, classification diagnostics, held-out metrics, confusion matrices, Gemini routing reports, and matching or duplicate review tables. A/B category labels are used for training/validation/test partitions; held-out test labels never enter model fitting or Gemini prompts. Spend labels are post-run evaluation only.

## Optional Gemini fallback

The classifier is local first. For A and B, the local TF-IDF/logistic model is trained from a deterministic training partition of that company's supplied labels. Validation selects its low-confidence threshold; a held-out test partition is reserved for comparison. For a new company such as C, there is no label-trained model, so the pipeline uses the taxonomy-only local classifier and can ask Gemini to resolve low-confidence cases.

The optional provider uses Google's `google-genai` SDK with model ID `gemini-3.8-flash`. Set `GEMINI_API_KEY` in the process environment to enable requests. The key is never stored in project config. Without it, the pipeline keeps local predictions and does not fail. A dry run reports bounded, deduplicated request and token/cost estimates without calling Gemini:

```powershell
python -m src.pipeline --all --output-dir output --name candidate --gemini-dry-run
```

Per-company thresholds, request budgets, timeout, retries, cache path, and current pricing assumptions are in the YAML `classification_engine` section. The SQLite cache key includes the model, transaction evidence, taxonomy candidates, and taxonomy levels. Gemini receives only the relevant fields for one unique uncertain transaction, not a dataset or row ID. Audit reports include routing outcomes and actual token usage when the API provides it.

## Configuration

Company configs define the company metadata, reporting currency, source datasets, required columns, semantic-to-source column mappings, taxonomy level names, and spend rules. A new company can be added by supplying the same sections plus its dataset paths and mappings. For a new single-source company, set `transaction_dataset`. Configure approved/cancelled/not-approved status values and clear non-procurement description patterns when the source supports them. `configs/company_C.yaml` is a template; replace its TODO mappings after the blind-test files arrive. See the A and B configs for working examples.

The runtime classifier uses a TF-IDF/logistic model for companies with label training data, with the taxonomy-token classifier as the zero-label fallback. Sentence Transformers can be enabled only from an already installed local model directory; no model is downloaded. Records without sufficient evidence remain on the local fallback or are routed to Gemini within the configured request budget. Confidence values are model estimates and are not calibrated probabilities.

## Data handling and tests

Task data is confidential and must not be committed. `.gitignore` excludes the supplied company data directories and the assessment brief. Keep the original EDA notebooks and source CSVs unchanged.

Run regression and end-to-end checks with:

```powershell
python -m unittest src.test_preprocessing src.test_matching src.test_pipeline -v
python src/test_ingestion.py
```

See `APPROACH.md` for method and known limitations, and `DECISIONS.md` for assumptions.
