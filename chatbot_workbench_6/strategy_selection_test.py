# Copyright (c) 2024 skrivov
import os
import json
import random
from typing import Sequence, Optional, TypedDict
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# Load API key from .env file
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini-2024-07-18")

# Determine the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full paths to the JSON files
strategies_path = os.path.join(script_dir, 'strategies.json')
data_path = os.path.join(script_dir, 'data.json')

# Load strategies.json
with open(strategies_path, 'r') as f:
    strategies_data = json.load(f)

conversation_strategies = strategies_data['conversation_strategies']

# Prepare the conversation strategies string
conversation_strategies_str = "\n".join(conversation_strategies)

# Load data.json
with open(data_path, 'r') as f:
    data = json.load(f)

# Define the GirlfriendResponse class using structured output
class GirlfriendResponse(TypedDict):
    message: str  # Girlfriend's response message
    emotional_reaction: int  # -1, 0, or 1
    justification: str  # Brief justification of the emotional reaction

# Define the state schema using TypedDict
class ConversationState(TypedDict):
    messages: Sequence
    girlfriend_emotional_reaction: Optional[int]
    girlfriend_justification: Optional[str]
    boyfriend_mood_detection: Optional[str]
    boyfriend_strategy_selection: Optional[str]
    boyfriend_response: Optional[str]
    girlfriend_message: Optional[str]

# Define the BoyfriendNode class
class BoyfriendNode:
    def __init__(self, llm, conversation_strategies_str):
        self.llm = llm
        self.conversation_strategies_str = conversation_strategies_str
        # Define the system message
        self.system_message = SystemMessage(
            content=(
                "You are an empathetic boyfriend. Your goal is to support your girlfriend through attentive listening and appropriate responses. "
                "Use the conversation history to understand her emotions and respond accordingly."
            )
        )
        # Define the prompts
        self.detect_mood_template = PromptTemplate(
            input_variables=["conversation_history"],
            template="""
Given the following conversation with my girlfriend, identify her current mood and emotions. Provide only the mood, do not elaborate.

Conversation:
{conversation_history}
"""
        )

        self.select_strategies_template = PromptTemplate(
            input_variables=["mood", "conversation_strategies"],
            template="""
My girlfriend is feeling {mood}. From the following conversation strategies, select one strategy using these instructions:

Select the most appropriate conversation strategy based on her detected mood and the context of our conversation.
List only the number corresponding to the selected strategy.
{conversation_strategies}
"""
        )

        self.boyfriend_response_template = PromptTemplate(
            input_variables=["conversation_history", "selected_strategy"],
            template="""
You are an empathetic boyfriend. Here is the conversation so far:
{conversation_history}

Using the following conversation strategy, craft a warm and engaging response that feels natural and personalized:

{selected_strategy}

Remember to be supportive, understanding, and express genuine emotion in your reply.
"""
        )

    def detect_mood(self, conversation_history):
        prompt = self.detect_mood_template.format(conversation_history=conversation_history)
        messages = [self.system_message, HumanMessage(content=prompt)]
        mood = self.llm.invoke(messages).content.strip()
        return mood

    def select_strategy(self, mood):
        prompt = self.select_strategies_template.format(
            mood=mood,
            conversation_strategies=self.conversation_strategies_str
        )
        messages = [self.system_message, HumanMessage(content=prompt)]
        strategy_response = self.llm.invoke(messages).content.strip()
        return strategy_response

    def generate_response(self, conversation_history, selected_strategy):
        # Extract the strategy text based on the selected number
        strategy_number = selected_strategy.strip()
        strategy_lines = self.conversation_strategies_str.split('\n')
        selected_strategy_text = ""
        for line in strategy_lines:
            if line.strip().startswith(strategy_number + '.'):
                selected_strategy_text = line
                break
        if not selected_strategy_text:
            selected_strategy_text = "Be supportive and understanding."

        prompt = self.boyfriend_response_template.format(
            conversation_history=conversation_history,
            selected_strategy=selected_strategy_text
        )
        messages = [self.system_message, HumanMessage(content=prompt)]
        response = self.llm.invoke(messages).content.strip()
        return response, selected_strategy_text

    def act(self, state: ConversationState):
        # Prepare conversation history
        conversation_history = ''
        for message in state['messages']:
            if isinstance(message, HumanMessage):
                conversation_history += f"Girlfriend: {message.content}\n"
            elif isinstance(message, AIMessage):
                conversation_history += f"Boyfriend: {message.content}\n"

        # Step 1: Detect mood
        mood = self.detect_mood(conversation_history)

        # Step 2: Select strategy
        selected_strategy = self.select_strategy(mood)

        # Step 3: Generate boyfriend response
        boyfriend_response, strategy_text = self.generate_response(conversation_history, selected_strategy)

        # Create AIMessage for the boyfriend's response
        boyfriend_message = AIMessage(content=boyfriend_response)

        # Append the message to the conversation history
        messages = state['messages'] + [boyfriend_message]

        print("Boyfriend:", boyfriend_response)

        # Return the updates
        return {
            'boyfriend_mood_detection': mood,
            'boyfriend_strategy_selection': selected_strategy,
            'boyfriend_response': boyfriend_response,
            'messages': messages,
        }

# Define the GirlfriendNode class
class GirlfriendNode:
    def __init__(self, llm):
        self.llm = llm
        # Improved formatting for the system message
        self.system_message = SystemMessage(
            content=(
                "You are a loving and communicative girlfriend. You are chatting with your boyfriend. "
                "Respond to his message appropriately, expressing your feelings and thoughts.\n\n"
                "Your response should be in the following JSON format:\n"
                "{\n"
                '  "message": "Your message",\n'
                '  "emotional_reaction": -1 or 0 or 1,\n'
                '  "justification": "Brief justification of your emotional reaction"\n'
                "}\n"
            )
        )
        # Create a structured LLM for Girlfriend's responses
        self.structured_llm = self.llm.with_structured_output(GirlfriendResponse)

    def act(self, state: ConversationState):
        # Collect the conversation history
        conversation_history = [self.system_message] + state['messages']

        # Generate the girlfriend's response
        try:
            parsed_response: GirlfriendResponse = self.structured_llm.invoke(conversation_history)
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
        messages = state['messages'] + [girlfriend_message]

        print("Girlfriend:", parsed_response['message'])
        print(f"Emotional Reaction: {parsed_response['emotional_reaction']}")
        print(f"Justification: {parsed_response['justification']}")

        # Return the updates
        return {
            'girlfriend_emotional_reaction': parsed_response['emotional_reaction'],
            'girlfriend_justification': parsed_response['justification'],
            'girlfriend_message': parsed_response['message'],
            'messages': messages,
        }

# Create the conversation graph
def create_conversation_graph(llm, conversation_strategies_str):
    # Create Boyfriend and Girlfriend agents
    boyfriend = BoyfriendNode(llm, conversation_strategies_str)
    girlfriend = GirlfriendNode(llm)

    # Create the state graph using the ConversationState schema
    graph = StateGraph(ConversationState)

    # Add nodes for Girlfriend and Boyfriend
    graph.add_node("Girlfriend", girlfriend.act)
    graph.add_node("Boyfriend", boyfriend.act)

    # Define conversation flow: Start -> Girlfriend -> Boyfriend -> Girlfriend -> End
    graph.add_edge(START, "Girlfriend")
    graph.add_edge("Girlfriend", "Boyfriend")
    graph.add_edge("Boyfriend", "Girlfriend")
    graph.add_edge("Girlfriend", END)

    # Compile and return the graph
    return graph.compile()

# Main function to run the conversation simulation
if __name__ == "__main__":
    # Outer loop: 5 simulations
    all_results = []
    for simulation_num in range(5):
        print(f"\n=== Simulation {simulation_num +1} ===\n")
        # Pick a random message from data.json
        initial_message_data = random.choice(data)
        girlfriend_message_content = initial_message_data['girlfriend_message']

        # Initialize the conversation state
        initial_gf_message = HumanMessage(content=girlfriend_message_content)
        state: ConversationState = {
            'messages': [initial_gf_message],
            'girlfriend_message': girlfriend_message_content,
            'girlfriend_emotional_reaction': 0,  # Assume neutral for initial message
            'girlfriend_justification': None,
            'boyfriend_mood_detection': None,
            'boyfriend_strategy_selection': None,
            'boyfriend_response': None,
        }

        print("Girlfriend (initial):", girlfriend_message_content)

        # Create the conversation graph with the LLM
        conversation_graph = create_conversation_graph(llm, conversation_strategies_str)

        # Run the conversation simulation
        state = conversation_graph.invoke(state)

        # Collect results for benchmarking
        simulation_result = {
            "simulation_num": simulation_num + 1,
            "girlfriend_initial_message": state.get('girlfriend_message'),
            "girlfriend_emotional_reaction": state.get('girlfriend_emotional_reaction'),
            "girlfriend_justification": state.get('girlfriend_justification'),
            "girlfriend_response": state.get('girlfriend_message'),
            "boyfriend_mood_detection": state.get('boyfriend_mood_detection'),
            "boyfriend_strategy_selection": state.get('boyfriend_strategy_selection'),
            "boyfriend_response": state.get('boyfriend_response'),
        }
        all_results.append(simulation_result)

    # Save the results to output.json for benchmarking
    # Construct the full path to the output file
    output_path = os.path.join(script_dir, 'output.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
