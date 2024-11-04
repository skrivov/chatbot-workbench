# Copyright (c) 2024 skrivov

from typing import Optional
from typing_extensions import TypedDict

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

# Define the state of the conversation using TypedDict for a structured representation
class EmpatheticBoyfriendState(TypedDict):
    conversation_strategies: str
    girlfriend_message: str
    mood: Optional[str]
    selected_strategy: Optional[str]  
    boyfriend_response: Optional[str]

# Initialize the language model with a specific configuration
model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

# Define the prompt template for detecting mood from the girlfriend's message
detect_mood_template = PromptTemplate(
    input_variables=["girlfriend_message"],
    template="""
    You are a sentiment analysis tool. Given the following message from my girlfriend, identify her current mood. 
    Provide only the mood as a single word without any additional text or explanation.

    Message:
    {girlfriend_message}
    """
)

# Define the prompt template for selecting a conversation strategy based on mood
select_strategies_template = PromptTemplate(
    input_variables=["girlfriend_message", "mood", "conversation_strategies"],
    template="""
    You are a conversation strategy selector. Based on the following message and detected mood, choose the most appropriate conversation strategy.

    Girlfriend's Message:
    {girlfriend_message}

    Detected Mood:
    {mood}

    Conversation Strategies:
    {conversation_strategies}

    Select the strategy by providing only the corresponding number without any additional text.
    """
)

# Define the prompt template for generating a response using the selected strategy
boyfriend_response_template = PromptTemplate(
    input_variables=["girlfriend_message", "selected_strategy"],
    template="""
    You are an empathetic boyfriend. Respond to your girlfriend's message using the selected conversation strategy.

    Girlfriend's Message:
    "{girlfriend_message}"

    Selected Strategy:
    {selected_strategy}

    Craft a warm and engaging response that feels natural and personalized. Be supportive, understanding, and express genuine emotion in your reply.
    """
)

# Define the function to detect the girlfriend's mood
def detect_mood(inputs):
    detect_mood_chain = detect_mood_template | model | StrOutputParser()
    mood = detect_mood_chain.invoke(inputs)
    return {"mood": mood.strip()}

# Define the function to select an appropriate conversation strategy
def select_strategy(inputs):
    select_strategy_chain = select_strategies_template | model | StrOutputParser()
    selected_strategy = select_strategy_chain.invoke(inputs)
    return {"selected_strategy": selected_strategy.strip()}

# Define the function to generate a boyfriend's response based on the strategy
def generate_boyfriend_response(inputs):
    # Create the boyfriend response chain
    boyfriend_response_chain = boyfriend_response_template | model | StrOutputParser()
    # Generate the boyfriend's response
    response = boyfriend_response_chain.invoke(inputs)
    return {"boyfriend_response": response.strip()}

# Build the state graph for the conversation flow
graph = StateGraph(EmpatheticBoyfriendState)
graph.add_node(detect_mood)
graph.add_node(select_strategy)
graph.add_node(generate_boyfriend_response)

# Define the edges for the graph to establish the flow between nodes
graph.add_edge(START, 'detect_mood')
graph.add_edge('detect_mood', 'select_strategy')
graph.add_edge('select_strategy', 'generate_boyfriend_response')
graph.add_edge('generate_boyfriend_response', END)

# Compile the graph into an executable application
app = graph.compile()

# Define the list of conversation strategies
conversation_strategies = [
    "1. Active Listening: Pay close attention to her words and feelings, and reflect them back to show understanding.",
    "2. Express Empathy: Acknowledge and validate her feelings, showing that you understand and care.",
    "3. Provide Reassurance: Offer comfort and support to alleviate her concerns or anxieties.",
    "4. Use Positive Affirmations: Encourage her with positive and uplifting statements.",
    "5. Ask Open-Ended Questions: Invite her to share more about her thoughts and feelings.",
    "6. Offer Assistance: Gently offer help or solutions if appropriate.",
    "7. Use Humor Carefully: Lighten the mood with appropriate humor, if suitable.",
    "8. Give Space: Recognize when she needs space and respect it.",
    "9. Share Affection: Express love and affection through words.",
    "10. Apologize Sincerely: If you've done something wrong, apologize sincerely.",
    "11. Be Patient: Allow her to express herself fully without rushing.",
    "12. Clarify Understanding: Ask for clarification if you're unsure about something she said.",
    "13. Share Similar Experiences: Relate to her by sharing your own experiences, if appropriate.",
    "14. Avoid Judgement: Listen without criticizing or judging her feelings.",
    "15. Focus on the Present: Keep the conversation in the present moment, avoiding bringing up past issues.",
    "16. Encourage Self-Expression: Encourage her to share her feelings openly.",
    "17. Acknowledge Achievements: Recognize and praise her accomplishments.",
    "18. Respect Her Perspective: Understand that her viewpoint is valid.",
    "19. Avoid Problem-Solving Unless Asked: Sometimes she may just want to be heard, not have her problems fixed.",
    "20. Maintain a Calm Tone: Keep your tone calm and soothing."
]

# Define a list of diverse girlfriend messages for testing
girlfriend_messages = [
    # Positive Achievement
    "I just got promoted at work! I'm so excited but also a bit nervous about the new responsibilities.",
    # Apology and Regret
    "I'm really sorry I forgot our anniversary. I feel terrible about it and want to make it up to you.",
    # Sharing Achievements
    "I completed my first marathon today! It was exhausting but incredibly rewarding.",
    # Seeking Advice
    "I'm struggling to decide between two job offers. One offers a higher salary, and the other has better growth opportunities. What should I do?",
    # Feeling Overwhelmed
    "I've been feeling really overwhelmed with all the deadlines at work. I don't know how to keep up.",
    # Dealing with Loss
    "My grandmother passed away last night, and I'm not sure how to cope with the grief.",
    # Expressing Loneliness
    "I've been feeling lonely lately, even when I'm surrounded by people.",
    # Humorous Situation
    "I tried baking a cake today, but it turned out looking like a disaster. At least it tasted okay!",
]

# Function to run the empathetic boyfriend cycle for a single message
def run_boyfriend_cycle(girlfriend_message, conversation_strategies_str):
    inputs = {
        "girlfriend_message": girlfriend_message,
        "conversation_strategies": conversation_strategies_str
    }
    
    try:
        # Execute the graph and get the result
        result = app.invoke(inputs)
        
        # Display the outputs
        print("--------------------------------------------------")
        print("Girlfriend's Message:\n", result.get('girlfriend_message'))
        print("Detected Mood:\n", result.get('mood'))
        print("\nSelected Strategy:\n", result.get('selected_strategy'))
        print("\nBoyfriend's Response:\n", result.get('boyfriend_response'))
        print("--------------------------------------------------\n")
    except Exception as e:
        print(f"An error occurred: {e}")

# Prepare the conversation strategies string by joining the list with newline characters
conversation_strategies_str = "\n".join(conversation_strategies)

# Run the cycle for each girlfriend message
for idx, message in enumerate(girlfriend_messages, start=1):
    print(f"=== Round {idx} ===")
    run_boyfriend_cycle(message, conversation_strategies_str)
