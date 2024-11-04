# Copyright (c) 2024 skrivov
import os
import random
from dotenv import load_dotenv

from langchain_core.messages import  HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.graph import END
from pydantic import BaseModel, Field
from typing import Optional

# Load API key from .env file for secure access to the OpenAI API
load_dotenv()

# Define the structured output schema for the LLM's response
class NextAgent(BaseModel):
    agent_name: str = Field(description="The full name of the agent that should speak next")

# Initialize the LLM and wrap it to use structured output for agent selection
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
structured_llm = llm.with_structured_output(NextAgent)

# Define the state of the conversation
class ConversationState(BaseModel):
    history: list = Field(default_factory=list)  # Holds conversation history
    last_speaker: Optional[str] = None           # Tracks the last speaker, allowing None
    next_speaker: Optional[str] = None           # Field to track the next speaker, allowing None
        
    
# Define an agent node class to represent each historical figure
class AgentNode:
    def __init__(self, name, description, other_agents):
        self.name = name
        self.description = description
        self.other_agents = other_agents
        self.system_message = self.create_system_message()

    def create_system_message(self):
        """
        Create a system message to guide the agent's behavior and context.
        """
        other_agent_names = " and ".join(self.other_agents)
        system_prompt = f"""
        You are {self.name}, a former President known for your {self.description}. Engage in a conversation with 
        the other former Presidents: {other_agent_names}. You should only respond on behalf of yourself and not 
        speak for the other Presidents. If the conversation has not yet started, initiate it by making a statement 
        or asking a question related to politics or your presidency. Always respond naturally and directly with 
        one or two lines, without trying to represent the opinions or voices of the other Presidents. Talk about 
        politics, making light-hearted jokes about Republicans, Democrats, and other presidents' jobs during their 
        tenures. Prefix your response with {self.name}:
        """
        return SystemMessage(content=system_prompt)

    def act(self, state):
        """
        Generate the agent's response based on the conversation state.
        """
        conversation_history = state.history
        response = llm.invoke([self.system_message] + conversation_history)
        response_content = response.content.strip()
        state.history.append(HumanMessage(content=response_content))
        print("------------------------")
        print(response_content)        
       
        # Update last_speaker to reflect the current speaker
        state.last_speaker = self.name
        return state

# Define a mediator node class to oversee agent interactions
class MediatorNode:
    def __init__(self, agent_names):
        self.agent_names = agent_names        
        self.system_message = self.create_system_message()

    def create_system_message(self):
        """
        Create a system message to guide the Mediator's behavior.
        """
        agent_list_str = ", ".join(self.agent_names[:-1]) + f", and {self.agent_names[-1]}"
        system_prompt = f"""
        You are a Mediator overseeing a conversation between historical figures: {agent_list_str}. Based on the 
        ongoing conversation, determine which of these agents should speak next. Only choose from the following 
        names: {", ".join(self.agent_names)}. Ensure that the conversation flows smoothly and that all agents have 
        a chance to speak. Output the full name of the agent that should speak next.
        """
        return SystemMessage(content=system_prompt)

    def mediate(self, state):
        """
        Decide which agent should speak next, based on the conversation state.
        """
        conversation_history = state.history
        last_speaker = state.last_speaker

        # Exclude the last speaker from the valid choices
        valid_choices = [name for name in self.agent_names if name != last_speaker]
        valid_choices_message = HumanMessage(
            content=f"Select the next speaker only from the following options: {', '.join(valid_choices)}."
        )
        
        full_history = [self.system_message] + conversation_history + [valid_choices_message]
        response = structured_llm.invoke(full_history)
        agent_name = response.agent_name.split()[-1]  # Extract the last name for simplicity

        # Ensure the selected agent is a valid choice
        if agent_name not in valid_choices:
            agent_name = random.choice(valid_choices)

        # Set the next speaker in the state
        state.next_speaker = agent_name

        print(f"Mediator selected: {agent_name}")
        return state

# Create and compile the conversation graph
def create_conversation_graph(agent_classes, mediator_class):
    # Create agents
    agents = {name: AgentNode(**details) for name, details in agent_classes.items()}
    mediator = mediator_class(agent_names=list(agents.keys()))

    # Create graph
    graph = StateGraph(ConversationState)

    # Add nodes for agents
    for name, agent in agents.items():
        graph.add_node(name, agent.act)

    # Add mediator node
    graph.add_node('Mediator', mediator.mediate)

    # Define a function to select the next agent based on mediator's decision
    def next_agent(state):
        if state.next_speaker is None:  # Default to Reagan if no next speaker is set
            return "Reagan"
        return state.next_speaker

    # Add conditional edges from the mediator to agents based on selection
    edge_mapping = {name: name for name in agents.keys()}
    graph.add_conditional_edges('Mediator', next_agent, edge_mapping)

    # Transition each agent to END after they speak
    for name in agents.keys():
        graph.add_edge(name, END)

    # Start by invoking the mediator
    graph.add_edge(START, 'Mediator')

    # Compile and return the graph
    return graph.compile()

if __name__ == "__main__":
    agent_classes = {
        "Reagan": {
            "name": "Ronald Reagan",
            "description": "wit, humor, and ability to tell great anecdotes",
            "other_agents": ["Nixon", "Carter"]
        },
        "Nixon": {
            "name": "Richard Nixon",
            "description": "complex personality and historical impact",
            "other_agents": ["Reagan", "Carter"]
        },
        "Carter": {
            "name": "Jimmy Carter",
            "description": "diplomatic skills and humanitarian efforts",
            "other_agents": ["Reagan", "Nixon"]
        }
    }

    mediator_class = MediatorNode
    conversation_graph = create_conversation_graph(agent_classes, mediator_class)

    # Initialize the conversation state
    state = ConversationState(history=[], last_speaker=None)

    # Run the simulation loop for a defined number of passes
    for i in range(5):  # Simulate 5 passes (iterations)
        print(f"--- Conversation Pass {i + 1} ---")
        state = conversation_graph.invoke(state)
