!pip install langchain_openai
!pip install langchain
dbutils.library.restartPython()

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
def generate_workflow_yaml(file_path: str) ->str:
    """Generate databricks workflow yaml file from a file"""
    prompt = f"""
    You are a data engineer. generate a workflow yaml file with the file path as
    {file_path}
    Only return workflow yaml. No explanation, no markdown.
    example job looks like this 
resources:
  jobs:
    Job_Name:
      name: Job_Name
      tasks:
        - task_key: Test
          spark_python_task:
            python_file: /Workspace/Users/
              Agents/test.py
          environment_key: Default
      queue:
        enabled: true
      environments:
        - environment_key: Default
          spec:
            client: "3"
      performance_target: PERFORMANCE_OPTIMIZED"""
    return llm.predict(prompt).strip()


response2 = response.strip('').replace('yaml','')
response3 = response2.strip('').replace('```','')
response3 = response3.replace("The generated Databricks workflow YAML file is as follows:\n", "")
with open("/Workspace/Users/test_workflow.yaml", "w") as f:
    f.write(response3)
print(response3)

# Register tools
tools = [generate_workflow_yaml]

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
  " use the file_path /Workspace/Users/generated_code.py and generate databricks workflow yaml.Do not stop the Spark Session")
print("Generate databricks yaml:\n")
