# Copyright (c) 2024 skrivov

import os
import operator
from typing import Annotated, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages  # Ensure messages are handled properly

# Load environment variables from .env file (e.g., OPENAI_API_KEY)
load_dotenv()

# Initialize the LLM with the specified model and API key
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")


class ConversationState(BaseModel):
    """
    Represents the state of the conversation, holding a sequence of AI messages.
    """
    messages: Annotated[Sequence[AIMessage], operator.add] = Field(default_factory=list)


class JimNode:
    """
    Represents Jim Halpert in the conversation graph.
    """

    def __init__(self, llm):
        """
        Initialize JimNode with the language model.
        """
        self.llm = llm
        self.system_message = self.create_system_message()

    def create_system_message(self):
        """
        Create a system message defining Jim's persona and context.
        """
        return SystemMessage(content=(
            "You are Jim Halpert from The Office. You love Pam Beesly, and you're trying to "
            "make things right after a minor argument. Be playful, charming, and use humor to "
            "lighten the mood. Prefix your response with 'Jim:'"
        ))

    def act(self, state: ConversationState):
        """
        Generate Jim's response based on the conversation state.
        """
        # Collect the conversation history and pass it to the LLM
        conversation_history = [self.system_message] + state.messages
        response = self.llm.invoke(conversation_history)
        response_content = response.content.strip()
        print("------------------------")
        print(response_content)
        # Add Jim's response to the conversation state
        return {"messages": [AIMessage(content=response_content)]}


class PamNode:
    """
    Represents Pam Beesly in the conversation graph.
    """

    def __init__(self, llm):
        """
        Initialize PamNode with the language model.
        """
        self.llm = llm
        self.system_message = self.create_system_message()

    def create_system_message(self):
        """
        Create a system message defining Pam's persona and context.
        """
        return SystemMessage(content=(
            "You are Pam Beesly from The Office. You are whimsical, playful, and a bit unpredictable, "
            "but you love Jim. Engage with him and react to his attempts to make up, adding some playful banter. "
            "Prefix your response with 'Pam:'"
        ))

    def act(self, state: ConversationState):
        """
        Generate Pam's response based on the conversation state.
        """
        # Collect the conversation history and pass it to the LLM
        conversation_history = [self.system_message] + state.messages
        response = self.llm.invoke(conversation_history)
        response_content = response.content.strip()
        print("------------------------")
        print(response_content)
        # Add Pam's response to the conversation state
        return {"messages": [AIMessage(content=response_content)]}


def create_conversation_graph(llm):
    """
    Create and compile the conversation graph involving Jim and Pam.

    Args:
        llm: The language model to use for generating responses.

    Returns:
        The compiled conversation graph.
    """
    # Create Jim and Pam agents
    jim = JimNode(llm)
    pam = PamNode(llm)

    # Create the state graph
    graph = StateGraph(ConversationState)

    # Add nodes for Jim and Pam with their respective actions
    graph.add_node("Jim", jim.act)
    graph.add_node("Pam", pam.act)

    # Define conversation flow: Start -> Jim -> Pam -> End
    graph.add_edge(START, "Jim")
    graph.add_edge("Jim", "Pam")
    graph.add_edge("Pam", END)

    # Compile and return the graph
    return graph.compile()


if __name__ == "__main__":
    # Create the conversation graph with the LLM
    conversation_graph = create_conversation_graph(llm)

    # Initialize the conversation state
    state = ConversationState()

    # Simulation loop: Simulate 3 exchanges between Jim and Pam
    for _ in range(3):
        state = conversation_graph.invoke(state)
