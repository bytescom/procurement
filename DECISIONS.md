# Decisions

- I count A's ERP goods receipts as the main purchase records. When a tool line is a confirmed match, I exclude the tool copy as `DUPLICATE`. A tool export repeated with the same invoice, supplier, PO, amount, month and description is counted once.
- For A's tool data, only configured approved or paid statuses are included. Cancelled rows use `CANCELLED`; rejected, on-hold, blank or unknown statuses use `NOT_APPROVED`.
- B has no approval status. I keep its invoice lines unless the account description clearly identifies an intercompany payable. Repeated invoice and line numbers are review candidates; I do not remove them automatically.
- For a new company, I can configure approved, cancelled and rejected statuses plus clear non-procurement description patterns. I will only add C exclusions after reviewing its source fields.
- A ERP amounts are already net. A tool amounts are gross, so I remove VAT using the row's rate and use 15% when it is blank. B amounts are converted to USD using the source rate or the matching monthly FX rate.
- Missing or invalid amounts stay missing and are reported. I do not estimate them. Three B lines have no source amount.
- I use supplier and item IDs when present. When they are missing, I use exact normalized names or descriptions with supplier and unit evidence. I avoid fuzzy auto-merges; different IDs for the same real supplier or item may therefore remain separate.
- For A matching, I require a unique pair and strong evidence such as matching supplier, net amount and PO, or matching supplier, amount and month. Fuzzy and split-receipt candidates are kept for review.
- I read each category tree from its CSV. Missing lower levels stay blank. For training, I only use labels that match the current transaction IDs. Fresh A2/B2 IDs will not match the original labels, so those runs use the taxonomy-only fallback unless a separate training-data workflow is added.
- Confidence is a model score, not a calibrated probability. Gemini is optional, receives selected transaction fields without record IDs or labels, and was not called during the latest run. Its cost figure is only a dry-run estimate.
- The C file layout and taxonomy are not present yet. `configs/company_C.yaml` is a template; its TODO mappings must be replaced with the real C headers before it can run. I will decide C-specific spend rules from the supplied file and record any assumptions here.
