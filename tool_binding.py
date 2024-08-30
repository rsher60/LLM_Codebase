from typing import Union, List
import re
from dotenv import load_dotenv
from langchain.agents import tool, create_structured_chat_agent , AgentType

from langchain_openai import ChatOpenAI

from langchain.tools import Tool

from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case

    return len(text)

@tool
def count_vowels(text:str) -> int:
  """Returns the count of vowels in a word"""
  counter = 0
  text = text.strip("'\n").strip(
      '"'
  )
  for l in text.lower():
    if l in ('a','e','i','o','u'):
      #print (l)
      counter += 1
  return counter


llm = ChatOpenAI()

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    """Finds a tool by its name."""
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

tools = tools = [get_text_length , count_vowels]
llm_with_tools = llm.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}

template = """
  Answer the following questions as best you can. You have access to the following tools:

  {tools}

  Use the following format:

  Question: the input question you must answer
  Thought: you should always think about what to do
  Action: the action to take, should be one of [{tool_names}]
  Action Input: the input to the action
  Observation: the result of the action
  ... (this Thought/Action/Action Input/Observation can repeat N times)
  Thought: I now know the final answer
  Final Answer: the final answer to the original input question

  Begin!

  Question: {input}
  Thought:
  """



def call_tools(msg: AIMessage) -> Runnable:
    """Helper function to call all requested tools sequentially and returns their outputs."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


chain =  llm_with_tools | call_tools

def main():
    question = "how many characters are there in the word 'DOG'? how many vowels are there in the word 'AMERICA'?"
    responses = chain.invoke(question)

    for i, response in enumerate(responses):
        question_text = question.split("?")[i]
        print(f"For question: '{question_text.strip()}', answer: {response['output']}")


if __name__ == "__main__":
    main()