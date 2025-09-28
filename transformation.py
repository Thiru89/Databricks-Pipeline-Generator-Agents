from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",   # or gpt-4o, gpt-4.1, gpt-3.5-turbo, etc.
    openai_api_key="",
    temperature=0.0,
    max_tokens=1000
)

from langchain.agents import initialize_agent
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
import pandas as pd

#define tools the agent can use
@tool
def preview_data(table: str) -> str:
    """Returns a preview of the data in the table."""
    df = spark.read.table(table).limit(10).toPandas()
    return df.head()

@tool
def suggest_transformations(column_summary: str) ->str:
    "This method suggests transformations based on column summary"
    prompt = f"""
    you are a data engineer assistant. based on the following column summary, suggest simple, short ETL transformation steps. but do not drop the columns
    output format : each suggestion on a new line, without explanations or markdown.
    Example: Remove $ from reveneue and cast to float Column summary:{column_summary}"""
    return llm.predict(prompt).strip()

@tool
def generate_spark_code(transform_description: str) ->str:
    """Generate Pyspark code from transformation description."""
    prompt = f"""
    You are a data engineer. Write a pyspark code to apply the following ETL transformations to Dataframe called 'df'.
    Transformations:
    {transform_description}
    Only return the Pyspark code and overwrite if the target table exists. No explanation, no markdown."""
    return llm.predict(prompt).strip()


@tool
def extract_only_code(code_value: str) ->str:
    """extract only code"""
    prompt = f"""
    You are a data engineer. extract the only the code from the response.
    example remove ```python or ``` from the code:
    {code_value}
    Only return the Pyspark code. No explanation, no markdown."""
    return llm.predict(prompt).strip()


tools = [preview_data, suggest_transformations, generate_spark_code, extract_only_code]

#create Agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    memory=ConversationBufferMemory()
)
#run the agent
response = agent.run(
  "Preview the table mycatalog.myschema.customer_sample and generate Spark code to read the table, clean it, and finally write the dataframe into a table called mycatalog.myschema.customer_sample_clean and extract only the code and make sure syntax is correct. Do not stope the Spark Session ")
print("Generate Pyspark Code:\n")
print(response)


response2 = response.strip('```').replace('python','')
print(response2)
exec(response2)
