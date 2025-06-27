# %%
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from IPython.display import display
import json
import os

#Connect to database
engine = create_engine("postgresql://joey@localhost:5432/openparliament2")
inspector = inspect(engine)
print(inspector.get_table_names(schema="public"))

# %%
for tbl in inspector.get_table_names(schema="public"):
    print(f"\n=== {tbl} ===")
    df = pd.read_sql(f"SELECT * FROM {tbl} LIMIT 5", engine)
    display(df)

# %%
#Helper code to extract tables with content relating to proceedings and analyze tables of interest 

tables_with_proceedings = [
    "bills_bill",
    "bills_billtext",
    "bills_votequestion",
    "hansards_statement"
]

#Helper function to investigate individual tables

def investigate_table(tbl):
    print(f"\n=== {tbl} ===")

    df = pd.read_sql(f"SELECT * FROM {tbl} LIMIT 3", engine)
    
    # list of column names
    print("Columns:")
    print(list(df.columns))
    
    # # column types
    # print("\nColumn dtypes:")
    # for col, dtype in df.dtypes.items():
    #     print(f" - {col}: {dtype}")
    
    # # show the sample rows
    # display(df)

for table in tables_with_proceedings:
    investigate_table(table)

#investigate_table(tables_with_proceedings[3])


# %%
columns_of_interest = {
    "bills_bill":['id','text_docid','name_en','status_code','law'],
    "bills_billtext":['bill_id','docid','text_en','summary_en'],
    "bills_votequestion":['id','bill_id','description_en','result'],
    "hansards_statement":['id','document_id','h1_en','h2_en','content_en'],
}

#Create new dataframes for tables we care about, with only columns of interest

for table, cols in columns_of_interest.items():
    cols_sql = ", ".join(f'"{c}"' for c in cols)
    sql = f'SELECT {cols_sql} FROM public."{table}"'
    df = pd.read_sql(sql, engine)
    # Dynamically assign to a variable like bills_bill_clean, etc.
    globals()[f"{table}_clean"] = df

# %%
#Check new dataframes

clean_dfs = [name for name in globals() if name.endswith('_clean')]

for df_name in clean_dfs:
    print(f"\n--- {df_name} ---")
    display(globals()[df_name].head(3))

# %%
# HANSARDS STATEMENT AGGREGATION

df_statement_per_document = (
    hansards_statement_clean
    .groupby('document_id')
    .apply(
        lambda grp: grp[['h1_en','h2_en','content_en']]
                   .to_dict(orient='records')
    )
    .reset_index(name='hansards_statements')
)

# %%
# BILLS AGGREGATION

df_docids_agg = (
    bills_billtext_clean
      .groupby('bill_id')['docid']
      .apply(list)
      .reset_index(name='text_docid')
)

df_texts_agg = (
    bills_billtext_clean
      .groupby('bill_id')
      .apply(lambda d: d[['docid','text_en','summary_en']]
                       .to_dict(orient='records'))
      .reset_index(name='bill_texts')
)

df_votes_agg = (
    bills_votequestion_clean
      .groupby('bill_id')
      .apply(lambda d: d[['id','description_en','result']]
                       .to_dict(orient='records'))
      .reset_index(name='vote_questions')
)

bills_master = (
    bills_bill_clean
      .rename(columns={'id':'bill_id'})
      .drop(columns=['text_docid'])
)

df_bills_merged = (
    bills_master
    .merge(df_docids_agg, on='bill_id', how='left')
    .merge(df_texts_agg,   on='bill_id', how='left')
    .merge(df_votes_agg,   on='bill_id', how='left')
)

# 3) Convert any missing entries into empty lists
for col in ('text_docid','bill_texts','vote_questions'):
    df_bills_merged[col] = df_bills_merged[col].apply(lambda x: x if isinstance(x, list) else [])

# 4) Preview
df_bills_merged.head()


# %%
final_dfs = ['df_bills_merged', 'df_statement_per_document']

# Show current working directory
print("Exporting to directory:", os.getcwd())

for df_name in final_dfs:
    # Retrieve the DataFrame object
    df = globals().get(df_name)
    if df is None:
        print(f"DataFrame '{df_name}' not found.")
        continue

    # Convert to list of dicts
    docs = df.to_dict(orient='records')
    
    # Filename to write
    filename = f'{df_name}.json'
    
    # Write out to JSON with pretty indentation
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    
    print(f"Written {len(docs)} records to {filename}")