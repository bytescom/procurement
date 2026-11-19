# Approach

I built one command to load the company files, clean the values, decide what to count, match likely duplicates, make supplier and item keys, and classify each included line. Company settings and column names are in YAML. The taxonomy is read from its CSV at run time.

## Data and spend

For A, I remove VAT from the invoice-tool gross amount and compare its lines with ERP receipts. I only mark a tool line as a duplicate when the matching evidence is strong; uncertain matches stay for review. ERP receipts are the canonical records. I use the configured invoice status to exclude cancelled, rejected and on-hold tool lines.

For B, I convert line amounts to USD using the source rate or the monthly FX file. There is no approval field, so I exclude only clear intercompany descriptions and keep repeated invoice numbers for review. Three B lines have no source amount; I leave their spend blank rather than inventing a value.

I prefer source supplier and item IDs. Where an ID is missing, I use exact normalized text evidence. I do not automatically merge fuzzy supplier or item matches. This can leave the same real entity under different source IDs.

## Classification

For A and B, I train a TF-IDF/logistic model on the labelled rows that match the current input records. I choose the confidence threshold on validation data and report results on a held-out test split grouped by identical description text. If there are no usable labels, I use taxonomy token matching. Gemini is an optional fallback for uncertain included lines; its responses are checked against the loaded taxonomy.

On the latest grouped held-out run, raw TF-IDF/logistic accuracy before the confidence threshold was A: 99.3%, 95.8%, 95.0% at L1-L3; B: 90.6%, 89.7%, 87.3%. The corresponding accepted coverage was 55.8% for A and 61.0% for B. The taxonomy-only classifier scored A: 34.0%, 4.8%, 2.8%; B: 46.6%, 45.9%, 27.0%. These are A/B held-out results, not a labels-hidden A result. I have not measured zero-label accuracy on A or accuracy on Company C.

The measured classification stage took about 91 seconds for A and 41 seconds for B on this machine. The latest dry run estimated about $0.073 total across 106,577 A/B rows, or about $0.00068 per 1,000 rows, with a 50-request cap per company. This is a prompt-based estimate; no Gemini API calls or actual API costs were measured.

## Limits

The Company C YAML is a template because the C files and column names are not in the current data folder. I will map it to the supplied invoice and taxonomy headers when those files are available. Fresh A2/B2 IDs do not match the original label IDs, so those rows use the no-label fallback rather than the trained A/B model. The three missing B amounts remain unknown.
