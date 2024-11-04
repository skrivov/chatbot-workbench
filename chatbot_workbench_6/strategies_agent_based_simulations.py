# Copyright (c) 2024 skrivov
import os
import json
import random
from typing import List, Optional, TypedDict
from dotenv import load_dotenv  # For loading environment variables from a .env file

# Import necessary modules from LangChain and LangGraph
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# Load the API key from the .env file
load_dotenv()

# Initialize the language model with the specified API key and model
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini-2024-07-18"
)

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full paths to the JSON files containing strategies and data
strategies_path = os.path.join(script_dir, 'strategies.json')
data_path = os.path.join(script_dir, 'data.json')

# Load conversation strategies from strategies.json
with open(strategies_path, 'r') as f:
    strategies_data = json.load(f)

conversation_strategies = strategies_data['conversation_strategies']

# Prepare the conversation strategies as a single string
conversation_strategies_str = "\n".join(conversation_strategies)

# Load initial messages from data.json
with open(data_path, 'r') as f:
    data = json.load(f)

# Define the GirlfriendResponse schema using TypedDict
class GirlfriendResponse(TypedDict):
    message: str                   # Girlfriend's response message
    emotional_reaction: int        # Emotional reaction: -1 (negative), 0 (neutral), or 1 (positive)
    justification: str             # Brief justification of the emotional reaction

# Define the conversation state schema using TypedDict
class ConversationState(TypedDict):
    messages: List[HumanMessage]           # List of conversation messages
    conversation_log: List[dict]           # Log of conversation steps

# Define the BoyfriendNode class responsible for the boyfriend's actions
class BoyfriendNode:
    def __init__(self, llm, conversation_strategies_str):
        self.llm = llm
        self.conversation_strategies_str = conversation_strategies_str

        # Define the system message for context
        self.system_message = SystemMessage(
            content=(
                "You are an empathetic boyfriend. Your goal is to support your girlfriend through attentive listening "
                "and appropriate responses. Use the conversation history to understand her emotions and respond accordingly."
            )
        )

        # Define the prompt templates
        self.detect_mood_template = PromptTemplate(
            input_variables=["conversation_history"],
            template="""
Given the conversation below, identify your girlfriend's current mood. Provide only a single word describing the mood.

Conversation:
{conversation_history}
"""
        )

        self.select_strategies_template = PromptTemplate(
            input_variables=["mood", "conversation_strategies"],
            template="""
Your girlfriend is feeling "{mood}". Based on the conversation strategies below, select the most appropriate one.

Consider the following guidelines:
- **Match the Mood**: Choose a strategy that aligns closely with her emotional state.
- **Contextual Relevance**: Ensure the strategy fits the specifics of her message.
- **Avoid Repetition**: Vary strategies to keep conversations engaging.
- **Emotional Support**: Prioritize strategies that offer the support she needs.
- **Encourage Positivity**: When appropriate, help shift her mood in a positive direction.

List only the number corresponding to the selected strategy.

Conversation Strategies:
{conversation_strategies}
"""
        )

        self.boyfriend_response_template = PromptTemplate(
            input_variables=["conversation_history", "selected_strategy_text"],
            template="""
You are an empathetic boyfriend. Here is the conversation so far:
{conversation_history}

Using the following conversation strategy, craft a warm and engaging response that feels natural and personalized:

{selected_strategy_text}

Remember to be supportive, understanding, and express genuine emotion in your reply.
"""
        )

    def _format_conversation_history(self, messages):
        """Formats the conversation history into a string."""
        conversation_history = ''
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation_history += f"Girlfriend: {message.content}\n"
            elif isinstance(message, AIMessage):
                conversation_history += f"Boyfriend: {message.content}\n"
        return conversation_history.strip()

    def act(self, state: ConversationState):
        """Performs the boyfriend's actions: detect mood, select strategy, and respond."""
        # Prepare conversation history
        conversation_history = self._format_conversation_history(state['messages'])

        # Step 1: Detect mood
        prompt = self.detect_mood_template.format(conversation_history=conversation_history)
        messages = [self.system_message, HumanMessage(content=prompt)]
        mood = self.llm.invoke(messages).content.strip()
        print(f"Detected Mood: {mood}")

        # Step 2: Select strategy
        prompt = self.select_strategies_template.format(
            mood=mood,
            conversation_strategies=self.conversation_strategies_str
        )
        messages = [self.system_message, HumanMessage(content=prompt)]
        selected_strategy_number = self.llm.invoke(messages).content.strip()
        print(f"Selected Strategy Number: {selected_strategy_number}")

        # Step 3: Generate boyfriend response
        boyfriend_response, selected_strategy_text = self.generate_response(conversation_history, selected_strategy_number)

        # Create AIMessage for the boyfriend's response
        boyfriend_message = AIMessage(content=boyfriend_response)

        # Append the message to the conversation history
        state['messages'].append(boyfriend_message)

        # Record the step in conversation_log
        state['conversation_log'].append({
            'speaker': 'Boyfriend',
            'message': boyfriend_response,
            'detected_mood': mood,
            'strategy_number': selected_strategy_number,
            'strategy_text': selected_strategy_text,
        })

        # Return the updated state
        return state

    def generate_response(self, conversation_history, selected_strategy_number):
        """Generates the boyfriend's response using the selected strategy."""
        # Extract the strategy text based on the selected number
        strategy_lines = self.conversation_strategies_str.split('\n')
        selected_strategy_text = next(
            (line for line in strategy_lines if line.strip().startswith(f"{selected_strategy_number}.")),
            "Be supportive and understanding."
        )

        prompt = self.boyfriend_response_template.format(
            conversation_history=conversation_history,
            selected_strategy_text=selected_strategy_text
        )

        messages = [self.system_message, HumanMessage(content=prompt)]
        response = self.llm.invoke(messages).content.strip()
        return response, selected_strategy_text

# Define the GirlfriendNode class responsible for the girlfriend's actions
class GirlfriendNode:
    def __init__(self, llm):
        self.llm = llm

        # Define the system message with clear instructions
        self.system_message = SystemMessage(
            content="""
You are a loving and communicative girlfriend chatting with your boyfriend. Respond to his message appropriately, expressing your feelings and thoughts.

Your response should be in the following JSON format:
{
  "message": "Your message",
  "emotional_reaction": -1 or 0 or 1,
  "justification": "Brief justification of your emotional reaction"
}
"""
        )

        # Create a structured LLM for the girlfriend's responses
        self.structured_llm = self.llm.with_structured_output(GirlfriendResponse)

    def _format_conversation_history(self, messages):
        """Formats the conversation history into a string."""
        conversation_history = ''
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation_history += f"Girlfriend: {message.content}\n"
            elif isinstance(message, AIMessage):
                conversation_history += f"Boyfriend: {message.content}\n"
        return conversation_history.strip()

    def act(self, state: ConversationState):
        """Generates the girlfriend's response based on the conversation."""
        # Prepare conversation history
        conversation_history = self._format_conversation_history(state['messages'])

        # Prepare prompt
        prompt = f"""
As the girlfriend, here is the conversation so far:
{conversation_history}

Respond to your boyfriend's last message. Provide your response in the following JSON format:
{{
  "message": "Your message",
  "emotional_reaction": -1 or 0 or 1,
  "justification": "Brief justification of your emotional reaction"
}}
"""

        messages = [self.system_message, HumanMessage(content=prompt)]

        # Generate the girlfriend's response
        try:
            parsed_response: GirlfriendResponse = self.structured_llm.invoke(messages)
        except Exception as e:
            print(f"Error parsing response: {e}")
            parsed_response = {
                'message': "I'm sorry, I didn't quite understand.",
                'emotional_reaction': 0,
                'justification': "I couldn't understand the message."
            }

        # Create HumanMessage for the girlfriend's response
        girlfriend_message = HumanMessage(content=parsed_response['message'])

        # Append the message to the conversation history
        state['messages'].append(girlfriend_message)

        # Record the step in conversation_log
        state['conversation_log'].append({
            'speaker': 'Girlfriend',
            'message': parsed_response['message'],
            'emotional_reaction': parsed_response['emotional_reaction'],
            'justification': parsed_response['justification'],
        })

        # Return the updated state
        return state

# Create the conversation graph
def create_conversation_graph(llm, conversation_strategies_str):
    """Creates and compiles the conversation graph for the simulation."""
    # Create Boyfriend and Girlfriend agents
    boyfriend = BoyfriendNode(llm, conversation_strategies_str)
    girlfriend = GirlfriendNode(llm)

    # Create the state graph using the ConversationState schema
    graph = StateGraph(ConversationState)

    # Add nodes for Boyfriend and Girlfriend
    graph.add_node("Boyfriend", boyfriend.act)
    graph.add_node("Girlfriend", girlfriend.act)

    # Define conversation flow: Boyfriend -> Girlfriend -> End
    graph.add_edge(START, "Boyfriend")
    graph.add_edge("Boyfriend", "Girlfriend")
    graph.add_edge("Girlfriend", END)

    # Compile and return the graph
    return graph.compile()

if __name__ == "__main__":
    # Create the conversation graph with the LLM
    conversation_graph = create_conversation_graph(llm, conversation_strategies_str)

    all_conversations = []

    # Outer loop: Run multiple simulations
    for simulation_num in range(5):
        print(f"\n=== Conversation {simulation_num + 1} ===\n")

        # Pick a random initial message from data.json
        initial_message_data = random.choice(data)
        girlfriend_message_content = initial_message_data['girlfriend_message']

        # Initialize the state with the girlfriend's initial message
        initial_gf_message = HumanMessage(content=girlfriend_message_content)
        state: ConversationState = {
            'messages': [initial_gf_message],
            'conversation_log': [{
                'speaker': 'Girlfriend',
                'message': girlfriend_message_content,
                'emotional_reaction': None,  # Initial message has no reaction
                'justification': None,
            }],
        }

        # Simulation loop: Simulate exchanges
        for _ in range(3):  # Simulate 3 exchanges
            state = conversation_graph.invoke(state)

        # Collect the conversation from 'conversation_log'
        conversation = state['conversation_log']

        # Append to all_conversations
        all_conversations.append({
            'simulation_num': simulation_num + 1,
            'conversation': conversation,
        })

        # Print the conversation
        print(f"\nConversation {simulation_num + 1}:")
        for entry in conversation:
            speaker = entry['speaker']
            message = entry['message']
            print(f"{speaker}: {message}")
            if speaker == 'Boyfriend':
                print(f"  Detected Mood: {entry.get('detected_mood')}")
                print(f"  Strategy Selected: {entry.get('strategy_number')}")
            elif speaker == 'Girlfriend' and entry['emotional_reaction'] is not None:
                print(f"  Emotional Reaction: {entry.get('emotional_reaction')}")
                print(f"  Justification: {entry.get('justification')}")

    # After all simulations, save the conversations to output.json
    output_path = os.path.join(script_dir, 'output.json')
    with open(output_path, 'w') as f:
        json.dump(all_conversations, f, indent=4)

    print("\nSimulation complete. Conversations saved to 'output.json'.")
