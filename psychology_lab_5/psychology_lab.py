# psychology_lab.py
# Copyright (c) 2024 skrivov

import os
import json
from typing import List, Dict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, ValidationError

# Import LangChain and LangGraph components
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from langgraph.graph import StateGraph, START, END

# Define the state class
class PsychologyLabState(TypedDict):
    strategies: List[str]
    data: List[Dict]
    messages: List[BaseMessage]
    iteration: int

# Initialize the language model
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

# Define Pydantic models for structured output
class DataEntry(BaseModel):
    girlfriend_message: str = Field(description="Sample message from the girlfriend")
    expected_strategy: str = Field(description="Number of the expected strategy as a string")

class AssistantResponse(BaseModel):
    strategies: List[str] = Field(
        description="List of at least 20 conversation strategies, each in the format 'number. **title**: description.'",
        min_length=20
    )
    data: List[DataEntry] = Field(
        description="List of at least 30 data entries with 'girlfriend_message' and 'expected_strategy'",
        min_length=30
    )

# Function to load initial files and state
def load_files() -> PsychologyLabState:
    state: PsychologyLabState = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'input')
    strategies_path = os.path.join(input_dir, 'strategies.json')
    data_path = os.path.join(input_dir, 'data.json')

    with open(strategies_path, 'r') as f:
        strategies_data = json.load(f)
    strategies = strategies_data['conversation_strategies']

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Initialize messages and iteration
    messages: List[BaseMessage] = []
    iteration = 0

    # Update state
    state['strategies'] = strategies
    state['data'] = data
    state['messages'] = messages
    state['iteration'] = iteration

    return state

# Node: Expert Node
expert_prompt = PromptTemplate(
    input_variables=["strategies", "data", "iteration"],
    template="""
You are an expert psychologist specializing in human interactions, empathy, personal psychology, and emotional intelligence.

We are developing data for training a boyfriend chatbot with high emotional intelligence. The goal of the chatbot is to keep the girlfriend happy. The chatbot should be supportive, offer genuine positive emotions, and foster the girlfriend's happiness and well-being.

The chatbot operates as follows:

1. It receives a **text message** from the girlfriend.
2. It identifies the emotions in the message.
3. Based on the emotions, it selects the appropriate communication strategy.
4. Using the selected strategy and the girlfriend's message, it generates a **text-based response**.

**Important Note**: The chatbot communicates **only through text messages**. Non-verbal communication methods (like facial expressions, gestures, or tone of voice) are not possible and should not be considered.

Your task is to review the provided strategies and data from a psychological standpoint. Ensure that:

- A wide range of **text-based communication strategies** is covered.
- The data provides scenarios necessary to invoke each strategy.
- Strategies are appropriate for text communication.

Identify any issues with coverage and correctness of strategy selection, and provide detailed suggestions for improvements. Focus on enhancing the strategies and data to better train the chatbot.

**Iteration**: {iteration}

**Strategies**:
{strategies}

**Data**:
{data}
"""
)

expert_chain = expert_prompt | llm

def expert_node(state: PsychologyLabState) -> PsychologyLabState:
    iteration = state['iteration']

    # Prepare content for the expert
    strategies = json.dumps(state['strategies'], indent=2)
    data = json.dumps(state['data'], indent=2)

    # Expert provides feedback
    expert_response = expert_chain.invoke({
        'strategies': strategies,
        'data': data,
        'iteration': iteration
    })
    expert_feedback = expert_response.content

    print(expert_feedback)

    # Add expert's feedback to messages
    state['messages'].append(AIMessage(content=expert_feedback))

    # Save feedback to 'work' folder
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work')
    os.makedirs(work_dir, exist_ok=True)
    feedback_path = os.path.join(work_dir, f'expert_feedback_iteration_{iteration}.txt')
    with open(feedback_path, 'w') as f:
        f.write(expert_feedback)

    return state

# Node: Assistant Node
assistant_prompt = PromptTemplate(
    input_variables=["expert_feedback", "strategies", "data"],
    template="""
You are an assistant psychologist specializing in human interactions, empathy, personal psychology, and emotional intelligence.

We are developing data for training a boyfriend chatbot with high emotional intelligence. The goal of the chatbot is to make the girlfriend happy. The chatbot should be supportive, offer genuine positive emotions, and foster the girlfriend's happiness and well-being.

The chatbot operates as follows:

1. It receives a **text message** from the girlfriend.
2. It identifies the emotions in the message.
3. Based on the emotions, it selects the appropriate communication strategy.
4. Using the selected strategy and the girlfriend's message, it generates a **text-based response**.

**Important Notes**:

- The chatbot communicates **only through text messages**. Non-verbal communication methods (like facial expressions, gestures, or tone of voice) are not possible and should not be included.
- **Your response must include at least 20 strategies and at least 30 data entries**.

You have received the following suggestions from the expert psychologist on improving the strategies and data files:

{expert_feedback}

Your task is to:

- Incorporate the expert's feedback into the strategies and data.
- Create new versions of strategies and data focused on **text-based communication**.
- Ensure that strategies are suitable for text messaging.

**Requirements**:

- Include **at least 20** text-based communication strategies.
- Each strategy should be numbered, with a short title in bold, and a description, formatted as 'number. **title**: description.'.
- The data should include **at least 30** sample text messages from the girlfriend, each matched with the expected strategy number.
- Remember that the goal of the chatbot is to be supportive, offer genuine positive emotions, and foster the girlfriend's happiness and well-being.

Provide the updated strategies and data in **JSON format** with keys 'strategies' and 'data'. Do not include any additional text or explanations. The 'strategies' should be a list of strings in the specified format, and 'data' should be a list of dictionaries with keys 'girlfriend_message' and 'expected_strategy'.
"""
)

# Create the assistant chain with structured output
structured_llm = llm.with_structured_output(AssistantResponse)
assistant_chain = assistant_prompt | structured_llm

def assistant_node(state: PsychologyLabState) -> PsychologyLabState:
    iteration = state['iteration']
    expert_feedback = state['messages'][-1].content if state['messages'] else ""
    strategies = json.dumps(state['strategies'], indent=2)
    data = json.dumps(state['data'], indent=2)

    # Assistant works on expert's suggestions
    try:
        assistant_response = assistant_chain.invoke({
            'expert_feedback': expert_feedback,
            'strategies': strategies,
            'data': data
        })
    except ValidationError as e:
        print(f"Validation Error in assistant response: {e}")
        # Optionally, retry or adjust the prompt
        # For now, we'll log the error and keep the previous strategies and data
        return state

    # Add assistant's response to messages
    state['messages'].append(HumanMessage(content=assistant_response.json()))
    
    print(assistant_response.json())
    # Update strategies and data from the assistant's response
    new_strategies = assistant_response.strategies
    new_data = [entry.dict() for entry in assistant_response.data]
    state['strategies'] = new_strategies
    state['data'] = new_data

    # Save new files to 'work' folder
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work')
    os.makedirs(work_dir, exist_ok=True)
    strategies_path = os.path.join(work_dir, f'strategies_iteration_{iteration}.json')
    data_path = os.path.join(work_dir, f'data_iteration_{iteration}.json')

    with open(strategies_path, 'w') as f:
        json.dump({'conversation_strategies': new_strategies}, f, indent=2)

    with open(data_path, 'w') as f:
        json.dump(new_data, f, indent=2)

    # Increment iteration
    state['iteration'] += 1
    print("Iteration: ", state['iteration'])

    return state

# Function to decide the next node
def should_continue(state: PsychologyLabState):
    max_iterations = 3
    if state['iteration'] >= max_iterations:
        return 'save_files'
    else:
        return 'expert_node'

# Node: Save Files
def save_files(state: PsychologyLabState) -> PsychologyLabState:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    strategies_path = os.path.join(output_dir, 'strategies.json')
    data_path = os.path.join(output_dir, 'data.json')

    with open(strategies_path, 'w') as f:
        json.dump({'conversation_strategies': state['strategies']}, f, indent=2)

    with open(data_path, 'w') as f:
        json.dump(state['data'], f, indent=2)

    return state

# Build the Graph
builder = StateGraph(PsychologyLabState)
builder.add_node('expert_node', expert_node)
builder.add_node('assistant_node', assistant_node)
builder.add_node('save_files', save_files)

builder.add_edge(START, 'expert_node')
builder.add_edge('expert_node', 'assistant_node')
builder.add_conditional_edges('assistant_node', should_continue)
builder.add_edge('save_files', END)

# Compile the graph
app = builder.compile()

# Initialize the state by loading files
initial_state = load_files()

# Run the application
app.invoke(initial_state)
