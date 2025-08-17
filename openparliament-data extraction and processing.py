import os
import re
import json
import string
import unicodedata
import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect

#Connect to database
engine = create_engine("postgresql://joey@localhost:5432/openparliament2")
inspector = inspect(engine)
print(inspector.get_table_names(schema="public"))

#1. DATA EXPLORATION 

for tbl in inspector.get_table_names(schema="public"):
    print(f"\n=== {tbl} ===")
    df = pd.read_sql(f"SELECT * FROM {tbl} LIMIT 5", engine)
    print(df)

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
    
for table in tables_with_proceedings:
    investigate_table(table)

#investigate_table(tables_with_proceedings[3])

columns_of_interest = {
    "bills_bill":['id','text_docid','name_en','status_code','law'],
    "bills_billtext":['bill_id','docid','text_en','summary_en'],
    "bills_votequestion":['id','bill_id','description_en','result'],
    "hansards_statement":['id','document_id','h1_en','h2_en','content_en','politician_id','time'],
    "core_politician":['id', 'name']
}

#Create new dataframes for tables we care about, with only columns of interest

for table, cols in columns_of_interest.items():
    cols_sql = ", ".join(f'"{c}"' for c in cols)
    sql = f'SELECT {cols_sql} FROM public."{table}"'
    df = pd.read_sql(sql, engine)
    # Dynamically assign to a variable like bills_bill_clean, etc.
    globals()[f"{table}_clean"] = df

#Check new dataframes

clean_dfs = [name for name in globals() if name.endswith('_clean')]

for df_name in clean_dfs:
    print(f"\n--- {df_name} ---")
    print(globals()[df_name].head(3))


#2. DATA AGGREGATION
#Helper function to normalize values
def normalize(o):
    if isinstance(o, datetime.datetime):
        return o.isoformat()
    if isinstance(o, dict):
        return {k: normalize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [normalize(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if pd.isna(o):
        return None
    return o

#2A. HANSARDS STATEMENT AGGREGATION

df_statement_per_document = (
    hansards_statement_clean
    .groupby('document_id')
    .apply(
        lambda grp: grp[['h1_en','h2_en','content_en','politician_id','time']]
                   .to_dict(orient='records')
    )
    .reset_index(name='hansards_statements')
)

#2B. BILLS AGGREGATION

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
    .merge(df_texts_agg, on='bill_id', how='left')
    .merge(df_votes_agg, on='bill_id', how='left')
)

#Convert any missing entries into empty lists
for col in ('text_docid','bill_texts','vote_questions'):
    df_bills_merged[col] = df_bills_merged[col].apply(lambda x: x if isinstance(x, list) else [])

df_bills_merged.head()


final_dfs = ['df_bills_merged', 'df_statement_per_document']


#Export final dataframes for bills

output_dir = os.getcwd()
os.makedirs(output_dir, exist_ok=True)

for df_name in final_dfs:
    # Retrieve the DataFrame object
    df = globals().get(df_name)
    if df is None:
        print(f"DataFrame '{df_name}' not found.")
        continue

    docs = df.to_dict(orient='records')
    clean_docs = [normalize(d) for d in docs]

    filename = os.path.join(output_dir, f'{df_name}.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(clean_docs, f, ensure_ascii=False, indent=2)

    print(f"Written {len(clean_docs)} records to {filename}")

# Show current working directory
print("Exporting to directory:", os.getcwd())


# 3. SEPARATION OF BILLS INTO INDIVIDUAL TEXT FILES

def create_safe_filename(name: str, max_length: int=100) -> str:
    """Make a filesystem-safe lowercase filename based on `name`."""
    if not isinstance(name, str):
        name = str(name) if pd.notna(name) else "unknown"
    # lowercase, replace whitespace with underscore
    name = name.lower().strip()
    name = re.sub(r"\s+", "_", name)
    # remove punctuation
    punct = re.escape(string.punctuation)
    name = re.sub(rf"[{punct}]+", "", name)
    # collapse multiple underscores and trim
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:max_length] or "unknown"


output_dir = "bills_json"
os.makedirs(output_dir, exist_ok=True)

# Iterate and write one JSON per bill
for idx, row in df_bills_merged.iterrows():
    # Build safe and unique filename
    safe_title = create_safe_filename(row["name_en"])
    filename = f"{row.bill_id}_{safe_title}.json"
    filepath = os.path.join(output_dir, filename)

    bill_record = {
        "bill_id":        row.bill_id,
        "name_en":        row.name_en,
        "status_code":    row.status_code,
        "law":            row.law,
        "text_docid":     row.text_docid,
        "bill_texts":     row.bill_texts,
        "vote_questions": row.vote_questions,
    }

    clean_record = normalize(bill_record)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(clean_record, f, ensure_ascii=False, indent=2)

    print(f"Wrote {filepath}")


# 4. AGGREGATION OF MP SPEECHES IN CHRONOLOGICAL ORDER

core_politician_clean['id'] = core_politician_clean['id'].astype(float) 

mp_statements = hansards_statement_clean.merge(
    core_politician_clean,
    left_on='politician_id',
    right_on='id',            
    how='inner'
)

mp_statements.drop(columns=['id_x','id_y'], errors='ignore', inplace=True)
mp_statements.head()

test_df = mp_statements.groupby('politician_id', sort=True)

mp_statements['time'] = pd.to_datetime(
    mp_statements['time'],
    errors='coerce',
    utc=True
)

mp_statements['time'] = pd.to_datetime(mp_statements['time'])

output = []

for pid, grp in mp_statements.groupby('politician_id', sort=True):
    grp = grp.sort_values('time')
    name = grp['name'].iloc[0]

    speeches = []
    for row in grp.itertuples(index=False):
        speeches.append({
            'document_id': row.document_id,
            'h1_en':       row.h1_en,
            'h2_en':       row.h2_en,
            'content_en':  row.content_en,
            'time':        row.time.isoformat()
        })
    
    record = {
        'politician_id': pid,
        'name':          name,
        'speeches':      speeches
    }
    
    output.append(normalize(record))

output_path = "politicians_speeches.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(output)} records to {output_path}")

## Exporting each MP's speeches into a different json file

mp_outdir = "mp_speeches_json"
os.makedirs(mp_outdir, exist_ok=True)

for pid, grp in mp_statements.groupby('politician_id', sort=True):
    grp = grp.sort_values("time")
    
    name = grp["name"].iloc[0] if len(grp) and pd.notna(grp['name'].iloc[0]) else "unknown"
    safe_name = create_safe_filename(name)

    try:
        if pd.notna(pid) and float(pid).is_integer():
            pid_str = str(int(pid))

        else:
            pid_str = str(pid)

    except Exception:
        pid_str = str(pid)


clean_record = normalize(record)

filename = f"{pid_str}_{safe_name}.json"
filepath = os.path.join(mp_outdir, filename)

# write one file per MP
with open(filepath, "w", encoding="utf-8") as f:
    json.dump(clean_record, f, ensure_ascii=False, indent=2)

# optional progress print
print(f"Wrote {filepath}")