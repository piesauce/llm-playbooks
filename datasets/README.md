**extract-data-openparliament.ipynb** contains the logic to extract data from **openparliament.ca/data-download**

The data exported is: 
- df_merged_bills.json: contains each unique bill, name of the bill, if it is the law, documents relating to the bill and text relating to each document
- df_statement_per_document.json: documents and their corresponding hansards statements
  
Please dump the file on the website into a database and update the path for the engine before running the code.
