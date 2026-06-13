# Step 1 : Extract schema
import os
import re
import pandas as pd
from sqlalchemy import create_engine,inspect,text
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langsmith import traceable
from dotenv import load_dotenv
import json
import sqlite3

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


db_url = "sqlite:///company_enterprise_large.db"

db = SQLDatabase.from_uri(
    "sqlite:///company_enterprise_large.db"
)

engine = create_engine("sqlite:///company_enterprise_large.db")

def extract_schema(db_url):
    engine = create_engine(db_url)

    inspector = inspect(engine)

    schema_text = ""

    for table in inspector.get_table_names():

        schema_text += f"\nTable: {table}\n"

        columns = inspector.get_columns(table)

        for col in columns:

            schema_text += (
                f"  - {col['name']} "
                f"({col['type']})\n"
            )

        pk = inspector.get_pk_constraint(table)

        if pk["constrained_columns"]:
            schema_text += (
                f"Primary Key: "
                f"{pk['constrained_columns']}\n"
            )

        fks = inspector.get_foreign_keys(table)

        for fk in fks:
            schema_text += (
                f"Foreign Key: "
                f"{fk['constrained_columns']} -> "
                f"{fk['referred_table']}"
                f"({fk['referred_columns']})\n"
            )

    return schema_text


# Step 2 : Text to SQL (Deepseek with Ollama)
@traceable
def text_to_Sql(schema,prompt):
    SYSTEM_PROMPT = """
    You are an expert SQLite SQL developer.Given a database schema and a user prompt, generate a valid SQl query that answer the prompt.
    Only use the tables and columns provided in the schema.Ensure the SQL syntax is correct and avoid using any unspported features.
    Output only the SQL as your response will be directly used to query the database..No preabmle please.

    IMPORTANT:

    - Generate ONLY SQLite SQL.
    - Never use QUALIFY.
    - Never use TOP.
    - Never use information_schema.
    - Never use backticks or markdown.
    - Never wrap output in ```sql.
    - For ROW_NUMBER(), RANK(), DENSE_RANK(), use a CTE or subquery and filter outside.
    - Return SQL only.

    Rules:

    1. Generate only SQLite SQL.
    2. If the user asks:
        - for each department name or id
        - per department id or name
        - by department id or name

    then use PARTITION BY.

    3. Use ROW_NUMBER() or RANK() for:
        - highest
        - second highest
        - nth highest

    4. Never use LIMIT/OFFSET for per-group ranking.

    5. Use column name which one user asked for display

    6. Return SQL only.

    Generate only SQLite SQL.
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system",SYSTEM_PROMPT),
        ("user","Schema:\n{schema}\n\nQuestion:{prompt}\n\nSQL Query")
    ])

    model = OllamaLLM(model="qwen2.5:7b")

    chain = prompt_template | model

    raw = chain.invoke({"schema":schema, "prompt": prompt})

    sql_query = re.sub(r"```sql|```", "", raw).strip()
    return sql_query


@traceable
def get_sql_query_from_user_input(user_input):
    schema = extract_schema(db_url)

    sql_query = text_to_Sql(schema,user_input)

    #print(sql_query)

    df = pd.read_sql_query(sql_query,engine)

    return df
    #return "Good Bye"

# Step 3 : Build the app and test the app
