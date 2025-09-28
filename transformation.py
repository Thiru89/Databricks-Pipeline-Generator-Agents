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
def scan_data(table: str) -> str:
    """scans the table to understand the metadata information"""
    df = spark.read.table(table).limit(100).toPandas()
    return df.head()

@tool
def transformation_recommendation(table_summary: str) ->str:
    "This method suggests required transformations recommendation"
    prompt = f"""
    You are a data engineer assistant. based on the following table summary, suggest simple, ETL transformation steps if required. but do not drop the columns
    output format : each suggestion on a new line, without explanations or markdown.
    Example: Remove blank spaces and cast to float Column summary:{table_summary}"""
    return llm.predict(prompt).strip()

@tool
def generate_pyspark_code(transform_dtls: str) ->str:
    """Generate a pyspark code from transformation recommendation"""
    prompt = f"""
    You are a data engineer. Write a pyspark code to apply the following basic transformations to Dataframe 'df'.
    Transformations:
    {transform_dtls}
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


tools = [scan_data, transformation_recommendation, generate_pyspark_code, extract_only_code]

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
  "scan or preview the table mycatalog.myschema.customer_sample and generate Spark code to read the table, cleanse it, and write the dataframe into a table called mycatalog.myschema.customer_sample_clean and extract only the code and make sure syntax is correct. Do not stope the Spark Session ")
print("Generate Pyspark Code:\n")
print(response)


response2 = response.strip('```').replace('python','')
print(response2)
exec(response2)
