# Decision

For A, if a purchase exists in both systems, I count the ERP entry and discard the version from the invoice tool, classifying it as a `DUPLICATE`. If the invoice, supplier, PO, amount, month, and description all match, I remove such duplicate data exported from the tool. If the match is not clear, I set it aside for review.

I exclude VAT from the amounts in A’s tool data (using the rate provided in the same row; if the rate is missing, I apply 15%). The ERP amounts are already net. For B, I use the USD exchange rate provided in the source data; if unavailable, I refer to the FX file for the relevant month. In three rows for B, no amount is provided in the source, so the spend remains blank.

An entry from the tool is counted only if its status is 'approved' or 'paid'. Entries marked as 'cancelled' are classified as `CANCELLED`, while those with a status of 'rejected', 'on-hold', blank, or unknown are classified as `NOT_APPROVED`. Since B lacks a status field, I retain its entries unless the description explicitly states 'intercompany payable'. The mere repetition of an invoice number is not sufficient to classify an entry as a 'duplicate'.

I primarily use the supplier and item ID for identification. If these are unavailable, I rely on the specific names or descriptions, while also considering the supplier and unit information for the item. I do not link records based solely on guesswork. Consequently, two distinct IDs might actually refer to the same supplier or item.

The taxonomy (classification) is derived from each company's CSV file. If a category lacks further sub-levels, I leave those levels blank. Labels are applied only when their record IDs match those in the current transaction file. Since new A2/B2 IDs will not correspond to existing labels, the 'no-label' option will be used for those entries.

The use of Gemini is optional. In my recent test, it failed to execute because no API key was configured. The cost indicated in the report is merely an estimate.

I have not yet received the invoice file or taxonomy for Company C; therefore, there is no output for Company C, and its configuration is not yet ready for use. The file `configs/company_C.yaml` serves as a template, with mappings still pending (TODO mappings). Once I receive the files, I will map the actual columns and establish spend rules for Company C based on the data.