# %%
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from IPython.display import display
import json
import os
import re
import string
import datetime
import numpy as np
import pandas as pd 

# %%
#Connect to database
engine = create_engine("postgresql://joey@localhost:5432/openparliament2")
inspector = inspect(engine)
print(inspector.get_table_names(schema="public"))

#DATA EXPLORATION 

for tbl in inspector.get_table_names(schema="public"):
    print(f"\n=== {tbl} ===")
    df = pd.read_sql(f"SELECT * FROM {tbl} LIMIT 5", engine)
    display(df)

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
    display(globals()[df_name].head(3))


#DATA AGGREGATION
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

# HANSARDS STATEMENT AGGREGATION

df_statement_per_document = (
    hansards_statement_clean
    .groupby('document_id')
    .apply(
        lambda grp: grp[['h1_en','h2_en','content_en','politician_id','time']]
                   .to_dict(orient='records')
    )
    .reset_index(name='hansards_statements')
)

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
    .merge(df_texts_agg, on='bill_id', how='left')
    .merge(df_votes_agg, on='bill_id', how='left')
)

# 3) Convert any missing entries into empty lists
for col in ('text_docid','bill_texts','vote_questions'):
    df_bills_merged[col] = df_bills_merged[col].apply(lambda x: x if isinstance(x, list) else [])

# 4) Preview
df_bills_merged.head()

# Show current working directory
print("Exporting to directory:", os.getcwd())

# %%
# AGGREGATION OF MP SPEECHES IN CHRONOLOGICAL ORDER

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


# %%
# %%
final_dfs = ['df_bills_merged', 'df_statement_per_document']


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
    
    # Write out to JSON
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
            default=lambda o: o.isoformat() if isinstance(o, datetime.datetime) else str(o)
        )

    
    print(f"Written {len(docs)} records to {filename}")

# %%
# SEPARATION OF BILLS INTO INDIVIDUAL TEXT FILES

def create_safe_filename(name: str, max_length: int = 100) -> str:
    name = name.lower().replace(" ", "_")
    punct = re.escape(string.punctuation)
    name = re.sub(rf"[{punct}]+", "", name)
    return name[:max_length]


# 3. Prepare output directory
output_dir = "bills_json"
os.makedirs(output_dir, exist_ok=True)

# 4. Iterate and write one JSON per bill
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


