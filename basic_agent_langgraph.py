import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from openai import OpenAI 
from langchain_openai import ChatOpenAI 
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END 


load_dotenv()

llm = "gpt-4o"

client = OpenAI()
model = ChatOpenAI(model=llm)

#Step 1 : Build a basic chat bot 


class State(TypedDict):
    messages: Annotated[list, add_messages]

def bot(state: State):
    print(state["messages"])
    return {"messages": [model.invoke(state["messages"])]}


graph_builder = StateGraph(State)

graph_builder.add_node("bot", bot)

# Step 3: Add an entry point to the graph 

graph_builder.set_entry_point("bot")
graph_builder.set_finish_point("bot")
graph = graph_builder.compile()

res = graph.invoke({"messages": ["hello, how are you?"]})

#print(res["messages"][-1].content)


while True:
    user_input = input("User: ")
    if user_input.lower() in ['q','quit','exit']:
        print("Goodbye")
        break
    for event in graph.stream({"messages":("user",user_input)}):
        for value in event.values():
            print("Assistant:" , value["messages"][-1].content)


