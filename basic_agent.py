import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

llm = "gpt-4o"

client = OpenAI()

response = client.chat.completions.create(
    model=llm , 
    messages=[
        {"role":"system","content":"You are a helpful assistant"},
        {"role":"user" , "content":"Who is Nelson mandela?"}
    ],
)

#print(response.choices[0].message.content)


#Create our own simple agent 

class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if system:
            self.messages.append({"role":"system","content": system})

    def __call__(self, message):
        self.messages.append({"role":"user","content":message})
        result = self.execute()
        self.messages.append({"role":"assistant","content":result})
        return result
    
    def execute(self):
        response = client.chat.completions.create(
            model=llm,
            messages=self.messages
        )
        return response.choices[0].message.content
    


prompt = """

You operate in a loop of THOUGHTS, ACTIONS, PAUSE and REFLECTION.
At the end of the loop, you will output an answer.
Use THOUGHTS to describe your thoughts about the question.
Use ACTIONS to describe your actions to solve the question.
Observation will be the result of running those ACTIONS.

Your available actions are:

calcualte: Calculate the answer
e.g calculate 4*7/3
Runs a calculation and return the numerical result - uses Python so be sure to use floating
point syntax if necessary 

planet_mass: Calculate the mass of a planet
e.g planet_mass: Earth
returns the mass of a planet in kg in the solar system 

Example Session:

Question: What is the combined mass of Earth and Jupiter?
Thought: I should find the mass of Earth and Jupiter
Action: planet_mass: Earth
PAUSE

You will be called again with this. 

Observation: The mass of Earth is 5.972 x 10^24 kg

You then output:

Answer: Earth has a mass of 5.972 X 10^24 kg

Next, call the agent again with:

Action: planet_mass: Jupiter

PAUSE

You then output:

Answer: Jupiter has a mass of 1.898 x 10^27 kg

Finally, calculate the combined mass:

Action: calculate 5.972 x 10^24 + 1.898 x 10^27

Observation: The combined mass of Earth and Jupiter is 1.953 x 10^27 kg

Answer: The combined mass of Earth and Jupiter is 1.953 x 10^27 kg

""".strip()
    

#Implement the function actions

def calculate(what):
    return eval(what)


def planet_mass(planet):
    masses = {
        "Mercury" : 0.33011,
        "Venus": 4.8675,
        "Earth": 5.97237,
        "Mars": 0.64171,
        "Jupiter": 1898.19,
        "Saturn": 568.34,
        "Uranus": 86.813,
        "Neptune": 102.413
    }
    return f"{planet} has a mass of {masses[planet]} x 10^24 kg"


known_actions = {"calculate": calculate, "planet_mass": planet_mass}

#Create the agent 
agent = Agent(system=prompt)

#response = agent("what is the mass of Earth?")
#print(response)

response = planet_mass("Earth")
print(response)