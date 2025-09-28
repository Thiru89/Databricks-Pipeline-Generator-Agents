# LangChain Databricks PySpark Code Generator Agent and Databricks pipeline Generator

This project demonstrates a LangChain Agent in Databricks that:

1. Scans the tables in Databricks.
2. provides recommendations for ETL transformations if required.
3. Generates PySpark transformation code.
4. Saves the generated code to Databricks Workspace (/Workspace/Shared) as a .py file.
5. It also generates a workflow Yaml file which then can be deployed via assert bundle.

## 🚀Features
- Multi-tool LangChain agent:
  - Table scan
  - Transformation recommendations
  - PySpark code generation
  - Code saving
  - generates a workflow yaml based on the file saved
- Automatic saving of generated code.
- Supports multi-input tools via LangChain's function calling.

## 📦 Prerequisites
- Databricks workspace
- Python environment with:
  - langchain
  - openai
  - pandas
- OpenAI API key stored in Databricks secrets.

## ⚙ Installation
```bash
%pip install openai langchain langchain-openai pandas
```

## Tools Defined
- `scan_data(table)`: Preview a Databricks table.
- `transformation_recommendation(column_dtls)`: Suggest ETL steps.
- `generate_pyspark_code(transform_dtls)`: Generate PySpark code.
- `save_to_workspace(filename, code)`: Save generated code.

## 💡Example Usage
```python
response = agent_executor.invoke({
    "input": "Preview the table 'mycatalog.myschema.customer', generate Spark code to clean it, and save as 'customer_clean.py'."
})
print(response)
```

## 📁 Folder Structure
```
.
├── agent.py
├── tools.py
├── README.md
```

