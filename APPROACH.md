# Method

I process files A and B using a single command. YAML files inform the loader about the location of each file and which columns to use. The output contains one row for every input line.

For file A, if an ERP receipt also appears in the invoice tool, I classify it as a `counted purchase`. I remove the VAT from the amounts in the tool and filter out confirmed duplicate entries. For file B, I convert the amounts into USD using either a specified rate or a monthly FX file. Since file B lacks an 'approval' field, I simply exclude entries explicitly identified as 'intercompany payables.' Duplicate invoice numbers are retained in the output to allow for subsequent verification.

Where supplier and item IDs are available, I utilize them. In the absence of IDs, I rely on text matching only if the text is an exact, normalized match. Cases involving ambiguity are left aside, as incorrect merges would be difficult to rectify. Consequently, a single actual supplier or item may be represented by multiple 'keys.'

The category tree is loaded from a taxonomy CSV file specific to each company. For both A and B, I train a TF-IDF/logistic regression model using labels derived from the existing input records. I use a `validation split` to determine the 'confidence threshold' and maintain a separate 'held-out split' for reporting scores. If no relevant label is found, the tool employs a 'taxonomy-text fallback'. While Gemini can be enabled to handle ambiguous entries, I have not performed any live API calls.

In a recent 'held-out run', the model's raw accuracy—prior to applying the threshold—was as follows: for A, 99.3%, 95.8%, and 95.0% at levels L1–L3; and for B, 90.6%, 89.7%, and 87.3%. Coverage above the threshold was 55.8% for A and 61.0% for B. Scores based solely on the taxonomy were 34.0%, 4.8%, and 2.8% for A; and 46.6%, 45.9%, and 27.0% for B. These are 'held-out' results. I have not tested A with all labels hidden, nor have I run the process for company C.

On this machine, the classification step took approximately 91 seconds for A and 41 seconds for B. Based on a 'dry run' (test run), the total cost for 106,577 rows is estimated at approximately $0.073, which translates to $0.00068 per 1,000 rows. There is a limit of 50 requests per company. This is merely an estimate; the actual cost for Gemini was $0, as I did not make any API calls.

## Company C

I have not received the invoice file or the taxonomy (classification structure) for Company C; therefore, I have not processed it, nor have I generated any output for it. `configs/company_C.yaml` is merely a template; its 'TODO' fields must be replaced with actual file names and column names. There are no labeled examples available for Company C, so I plan to use my own tree structure along with an 'unlabeled classifier,' with Gemini serving as a fallback option. Once I have reviewed the fields for Company C, I will establish 'spend rules' for it. In three rows for Company C, the 'source amount' was not provided; consequently, I have left the values ​​for those expenses blank.