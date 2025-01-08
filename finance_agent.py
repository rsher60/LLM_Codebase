import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
import json
from langgraph.graph import StateGraph , END
from typing import TypedDict , Annotated 
import operator 
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage 
from langgraph.checkpoint.sqlite import SqliteSaver
import pandas as pd
from io import StringIO
from tavily import TavilyCient
from typing import TypedDict, List 
from langchain_core.pydantic_v1 import BaseModel
#declare memory 

memory = SqliteSaver.from_conn_string(":memory:")

#Load environement variables 
load_dotenv()

tavily = os.get_env("TAVILY_API_KEY")

llm = "gpt-4o"

model = ChatOpenAI(model=llm)

class AgentState(TypedDict):
    task: str
    competitors: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str
    comparision: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int 
    max_revisions: int 

class Queries(BaseModel):
    queries: List[str]


#Define prompts for each node - IMPROVE as NEEDED
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Gather the financial data for the given company.Provide the detailed financial data"""
ANALYZE_DATA_PROMPT="""You are an expert financial analyst.Analyse the provided financial data and provide detailed insights and analysis."""
RESEARCH_COMPETITORS_PROMPT="""You are a researcher tasked with providing information about similar companies for performance comparision.Generate a list of search queries to gather relevant information. Only generate 3 queries max"""
COMPLETE_PERFORMANCE_PROMPT="""You are an expert financial analyst. Compare the financial performance of the given company with its competitors based on the provided data.
***MAKE SURE TO INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISION.***
"""
FEEDBACK_PROMPT="""You are a reviewer.Provide detailed feedback and critique for the provided financial comparision report."""
WRITE_REPORT_PROMPT = """You are a financial report writer.Write a comprehensive financial report based on the analysis, competitor research, comparision and feedback provided."""
RESEARCH_CRITIQUE_PROMPT="""You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""


def gather_financials_node(state: AgentState):
    #read the csv file into a pandas DataFrame
    csv_file = state["csv_file"]
    df = pd.read_csv(StringIO(csv_file))

    #Convert the Dataframe to a string
    financial_data_str = df.to_string(index=False)

    #Combine the financial data string with the task
    combined_content = (
        f"{state['task']}\n\n Here is the financial data:\n\n{financial_data_str}"

    )

    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content)
    ]

    response = model.invoke(messages)
    return {"financial_data":response.content}



def analyze_data_node(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state['financial_data']),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content}


def research_competitors_node(state: AgentState):
    content = state["content"] or []
    for competitor in state["competitors"]:
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
                HumanMessage(content=competitor)
            ]
        )

        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
    return {"content": content}