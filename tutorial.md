# Chatbot Workbench: Developing and Testing Chatbots using LangGraph and Agent Based Simulation

## Introduction

Testing chatbots can be challenging, especially when attempting to accurately simulate how the bot will perform in real-world scenarios. This tutorial introduces an innovative approach: using agent-based simulations to test chatbot behavior. By creating controlled environments in which chatbot agents interact, we can observe and analyze the efficiency and correctness of the responses produced.  We will use  LangGraph, a LangChain based  framework  designed to work with graph-based workflows, making it easier to create and manage conversational agents and create Agent Based Simulations. 

The first two sections of this tutorial introduce the key concepts essential for understanding the LangGraph architecture. Having struggled to learn LangGraph myself, I have made every effort to identify and consolidate the most crucial ideas in one place. By mastering these key concepts, you will begin to think in LangGraph. This will enable you to easily envision and design your own LangGraph applications.

 The third section adapts elements of the SELF-DISCOVER framework for simulating personal communications, focusing specifically on the problem of emotional intelligence. Our adaptation replaces the concept of reasoning modules with the concept of conversation strategies, enabling the chatbot to guide users through their emotions in a supportive manner. The fourth section describes the usage of the Retrieval-augmented generation or RAG pattern for extracting a list of conversation strategies from the documents. The fifth section demonstrates REFLECTION pattern and explains how a team of interacting LLM powered agents can  generate conversation strategies through gradual refinement. Finally, the last section demonstrates how various elements developed in the tutorial can be put together to test and refine chatbot prompts using different simulation environments. 

 This tutorial focuses on using LangChain and LangGraph to develop Agent Based Simulations. However, if you are interested only in Agent Based Simulation with vanilla LLMs, check out [my previous tutorial](https://github.com/skrivov/llm-powered-multiagent-simulation-tutorial). 

 
## Agent Based Models in LangGraph

**File**: [jim\_and\_pam\_1.py](./jim_and_pam_1.py)

### Creating Agents

Imagine simulating a conversation between Jim and Pam from *The Office*. To give each agent a distinct personality, we use LLM's  system prompt that's crafted and provided to the LLM before any conversation begins. In the `PamNode` class, for example, the `create_system_message()` method defines Pam Beesly's persona. It paints her as whimsical, playful, and deeply in love with Jim—guiding her responses to stay true to her character throughout the conversation.

The system prompt plays a key role in defining how Pam behaves during interactions, setting her personality and tone right from the start. In our simulation, Jim and Pam are LLM-powered agents, interacting with each other to recreate the witty and playful dynamics typical of their on-screen personas. By carefully designing their system prompts, we ensure that the conversations remain true to Jim and Pam's characters, making the interactions engaging and believable.

### Creating the Graph Structure

We construct our Agent Based simulation using LangGraph. Central to LangGraph is the computation state object, represented in this case by the `ConversationState` class. Each node in the graph is a function that takes the state of the graph as input and produces an update to the state.  Computation in LangGraph involves a series of updates to this state, carried out through calls to different nodes. The flow of these computations is defined by the graph's edges, which determine how one node transitions to the next. Typically, a node is expected to return either an object of the same type as the graph's state, (which in our case is `ConversationState`) or a dictionary that specifies an update to the state. If the state contains a sequence of messages, we can use the `Annotated` class and operators like `operator.add` to specify the type of update to the sequence. For instance, in the `PamNode` class, the `act()` method only returns updated messages, making the state changes more focused and efficient.

```python

# This class defines the  state of our StateGraph. It keeps the history of the conversation
# The annotation add_messages define the mannor in which the sequence in the field messages is updated

class ConversationState(BaseModel):
    messages: Annotated[Sequence[AIMessage], operator.add] = Field(default_factory=list)

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
```

The above code defines the conversation graph, laying out the sequence in which Jim and Pam interact. The nodes and edges represent the flow of the conversation, starting with Jim and ending after Pam speaks.

### Looping Through the Conversation

The simulation loop pattern presented here addresses the issue of recursion depth limits when looping through conversation cycles in LangGraph. Recursive loops that include a 'should\_continue' condition and run for over 20 iterations can hit the recursion depth limit, making them unsuitable for extended use. A better approach is to use an explicit iterative loop structure. In agent-based simulations, we can simplify the execution cycle into a straightforward loop, repeatedly calling the conversation graph in a cycle. This avoids recursion and prevents issues related to recursion depth limits or requiring predefined exit conditions.

```python
if __name__ == "__main__":
    # Create the conversation graph with the LLM
    conversation_graph = create_conversation_graph(llm)

    # Initialize the conversation state
    state = ConversationState()

    # Simulation loop: Simulate 3 exchanges between Jim and Pam
    for _ in range(3):
        state = conversation_graph.invoke(state)
```

This loop runs the entire cycle defined in the conversation graph multiple times. After the `invoke()` method executes the complete cycle,  the resulting state is passed to the next iteration, allowing the conversation to progress naturally while retaining the history of all exchanges in the `ConversationState` object.

### Runnable Interface

In our example, both the language model (`llm`) and the instance of the conversation graph are invoked using the `invoke()` method. This is part of LangChain's Runnable interface, which standardizes how custom chains are created and executed. Many LangChain components, such as chat models, LLMs, output parsers, and retrievers, use this protocol, making integration straightforward and efficient.

The standard interface includes several useful methods, such as `invoke()` for calling a chain on a single input, `batch()` for processing a list of inputs, and `stream()` for streaming back parts of a response. Each of these methods also has asynchronous versions—`ainvoke()`, `abatch()`, and `astream()`—that can be used to handle multiple requests concurrently.

### Summary of LangGraph Architecture

- **Computation State**: The core of LangGraph is the computation state, represented by the `ConversationState` class in our example.
- **Computation Sequence**: Computation involves sequential updates to the state, executed by different nodes. The sequence of these computations is specified by the edges of the graph, defining the flow from one node to the next.
- **Node Inputs and Outputs**: Nodes take entire graph's state as an input and typically return an object that specifies an update to the graph's state . With the `Annotated` class and operators like `operator.add`, nodes can return  updates to sequences contained in graph's state , making modifications more modular and efficient.
- **Runnable Interface**: Both the language model (`llm`) and the conversation graph instance are invoked using the `invoke()` method. This is part of LangChain's Runnable interface, which standardizes chain creation and execution.

## Mediator Pattern: Modeling Group Conversation

**File**: [group\_conversation\_2.py](./group_conversation_2.py)

Handling conversations between more than two participants in multi-agent simulations can get tricky. When three agents take turns in a strict cycle, the dialogue often feels forced and unrealistic. To solve this, we introduce a Mediator object—an LLM-based node that determines who speaks next, based on the conversation's flow. In our example, three specific presidents—Ronald Reagan, Richard Nixon, and Jimmy Carter—are having a conversation in a restaurant.

The conversation graph is set up with the Mediator acting as one of the agents, using conditional edges to dynamically guide the conversation based on the state. The MediatorNode class system prompt instructs the Mediator on how to handle interactions and select the next speaker.  

The Mediator node is responsible for ensuring smooth agent interactions by deciding who should speak next based on the current conversation state. Here is the conversation state of our graph:

```python
# Define the state of the conversation
class ConversationState(BaseModel):
    history: list = Field(default_factory=list)  # Holds conversation history
    last_speaker: Optional[str] = None           # Tracks the last speaker, allowing None
    next_speaker: Optional[str] = None           # Field to track the next speaker, allowing None
```

The mediator plays an active role in managing the conversation flow by analyzing the conversation history (`state.history`) and the last speaker (`state.last_speaker`). In the `mediate()` method, the mediator assembles a complete conversation history, including the system message and valid choices, and then invokes the LLM to decide the next speaker. After selecting the next speaker, the mediator updates the `ConversationState` (`state.next_speaker = agent_name`). This ensures that all participants get a chance to speak, keeping the conversation balanced and engaging.

For computation flows that depend on conditions within the state, LangGraph uses conditional edges. These edges determine which node(s) to transition to by calling a function. In the `create_conversation_graph()` function, conditional edges enable the mediator to dynamically choose the next speaker based on the context of the conversation. The `next_agent(state)` function helps determine who should speak next, defaulting to Reagan if `next_speaker` is `None`. This keeps the conversation fluid, ensuring agents take turns naturally.

```python
# Create and compile the conversation graph
def create_conversation_graph(agent_classes, mediator_class, steps, start_agent):
    # Create agents
    agents = {name: AgentNode(**details) for name, details in agent_classes.items()}
    mediator = mediator_class(agent_names=list(agents.keys()), max_steps=steps)

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
```

The Mediator pattern depends heavily on conditional edges, which play a crucial role in deciding the flow of conversation and ensuring a natural, responsive interaction among agents.

## Adapting Self-Discover Framework to Modeling Emotional Intelligence

**File:** [ empathetic\_boyfriend\_3.py](./empathetic_boyfriend_3.py)

Originally, the Self-Discover algorithm was introduced  in paper 'Self-Discover: Large Language Models Self-Compose Reasoning Structures' by Pei Zhou et al.. The core idea of SELF-DISCOVER is that LLMs autonomously select and arrange atomic reasoning modules, such as critical thinking or step-by-step analysis, to create a coherent reasoning structure for tackling complex problems. This approach significantly improves the efficiency and performance of LLMs on various benchmarks, including BigBench-Hard and MATH, with better results and reduced computational requirements compared to traditional methods like Chain of Thought (CoT).

We will adapt elements of  SELF-DISCOVER framework for simulating personal communications, specifically focusing on the problem of emotional intelligence. In our adaptation to the domain of personal empathetic conversations, the Self-Discover algorithm's atomic reasoning modules become conversation strategies. For example, atomic reasoning modules like 'critical thinking' or 'step-by-step analysis' become conversational strategies such as 'Active Listening' or 'Express Empathy.' These strategies are selected based on the user's emotional state and context, allowing the chatbot to guide the user through their feelings in a natural and supportive way.

The Self-Discover pattern, adapted for the domain of personal communications, follows these streamlined steps:

- **Mood Detection**: Analyze the user's message to determine their current mood.
- **Strategy Selection**: Based on the detected mood, choose the most appropriate conversational strategy to provide empathetic support.
- **Response Generation**: Craft a supportive response using the selected strategy.

The state graph here is built with distinct nodes for mood detection, strategy selection, and response generation, providing an adaptable framework for conversations.

As before, the important information is kept in a state object. However, this time we use a TypedDict object instead of Pydantic models, highlighting the versatility of LangGraph in defining conversation states. The state of the conversation is defined as follows:

```python
# Define the state of the conversation using TypedDict for a structured representation
class EmpatheticBoyfriendState(TypedDict):
    conversation_strategies: str
    girlfriend_message: str
    mood: Optional[str]
    selected_strategy: Optional[str]  
    boyfriend_response: Optional[str]
```

In this implementation we introduce two new elements of LangChain PromptTemplates and pipes. Prompt templates are similar to Python's F-strings (Formatted String Literals). Prompt Templates help to translate user input and parameters into instructions for a language model. Prompt templates take as input a dictionary, where each key represents a variable in the template to fill in. Here is an example of prompt template:

```python
# Define the prompt template for detecting mood from the girlfriend's message
detect_mood_template = PromptTemplate(
    input_variables=["girlfriend_message"],
    template="""Given the following message from my girlfriend, identify her current mood and emotions. Provide only the mood, do not elaborate.

Message:
{girlfriend_message}
"""
)
```

When we call this prompt template we need to pass the value to variable girlfriend\_message. The template then will insert it into string template and output it combined string.   When we use LangGraph, the value of the variable can be extracted from input to the node which is the state of the graph.

Another new element we introduce here is LangChain Expression Language (LCEL). LCEL enables the creation of complex pipes,  using the '|' operator.  These pipes are similar to Unix pipes. LCE pipes  allow seamless chaining of components like models and templates, making conversational workflow creation intuitive and efficient.

The following example creates a pipe that chains together detect\_mood\_template , LLM model and string output parser. 

```python
# Define the function to detect the girlfriend's mood
def detect_mood(inputs):
    detect_mood_chain = detect_mood_template | model | StrOutputParser()
    mood = detect_mood_chain.invoke(inputs)
    return {"mood": mood.strip()}
```

The function detect\_mood() takes the entire EmpatheticBoyfriendState object as input. It  passes it to the detect\_mood\_template, which extracts from it  the girlfriend\_message uses it to create  a prompt, which it sends it to the LLM model. The output from the LLM model is then processed by StrOutputParser(). The function returns update to the state of the graph. We see the Runnable interface in action once again. After creating the detect\_mood\_chain, it is invoked using the invoke() method, which is part of the Runnable interface.


We will use the empathetic boyfriend agent architecture developed here in subsequent sections of the tutorial. The list of strategies plays a crucial role in defining the agent's behavior. Moving forward, we will modularize this program by separating the list of strategies from the code and storing them in a JSON file. Separation of prompts and code will make the agent prompts more flexible and easily adjustable. As we shall see, it will also allow to test strategy prompts in different environments.


## Extracting Structured Output from Documents

**File**: [strategies\_from\_documents\_4/web\_based\_loader.py](./strategies_from_documents_4/web_based_loader.py)

Assume that you wish to extract a list of communication strategies from a set of documents providing recommendations on how to better communicate with your girlfriend and thus transfer the expertise contained in the documents to your empathetic boyfriend chatbot. Assume that, for each strategy, you also wish to have some data entries containing examples of messages from a girlfriend along with the number of the strategy that should be used in response to the message. The strategies and the data entries are written in a specific format. However, LLM responses often break guidelines on the response format. Thus, one challenge that needs to be addressed here is ensuring that the information extracted from the documents follows a specific format.

The following code fragments define the required output format from the model and also describe the format of the assistant's response, which combines multiple responses from the LLM into a single cohesive JSON file.

```python
# Define Pydantic models for structured output
class DataEntry(BaseModel):
    girlfriend_message: str = Field(description="Sample message from the girlfriend")
    expected_strategy: str = Field(description="Number of the expected strategy as a string")

# Format of the output of one call to LLM
class ProblemResponse(BaseModel):
    strategies: List[str] = Field(
        description="List of communication strategies, each in the format 'number. **title**: description.'"
    )
    data: List[DataEntry] = Field(
        description="List of data entries with 'girlfriend_message' and 'expected_strategy'"
    )

# Format of the final combined output
class AssistantResponse(BaseModel):
    strategies: List[str] = Field(
        description="List of at least 20 conversation strategies, each in the format 'number. **title**: description.'",
        min_length=20
    )
    data: List[DataEntry] = Field(
        description="List of at least 30 data entries with 'girlfriend_message' and 'expected_strategy'",
        min_length=30
    )

# Initialize the language model
llm = ChatOpenAI(temperature=0)

# Initialize the language model with structured output for ProblemResponse
structured_llm_problem = llm.with_structured_output(ProblemResponse)
```

The `structured_llm_problem` is invoked as a regular LLM with an appropriate prompt and will return output in the specified format.

The key part of the program is an implementation of the Retrieval-Augmented Generation (RAG) pattern, which involves three main steps: document retrieval, context generation, and response generation.

First, the document retrieval step uses document loaders, such as the `WebBaseLoader`, to fetch data from web-based sources. The raw text from these sources is then processed using the `RecursiveCharacterTextSplitter`, which splits the text into manageable chunks of 2000 characters with an overlap of 100 characters. This ensures that each chunk retains context from neighboring sections. The chunks are then stored in an in-memory vector store, implemented using `InMemoryVectorStore` with embeddings created by `OpenAIEmbeddings`. This allows efficient semantic searching of the document content.

In the document retrieval step, when a specific problem or prompt is provided, the vector store is queried for similar content. This is achieved by calling the `similarity_search()` method on the vector store, which retrieves the most relevant document chunks based on the problem description. The retrieved chunks are then combined to form a coherent context, which serves as background information for the language model to use during response generation.

The process of generating conversation strategies and mapping them to data entries is accomplished within the `process_problems` function. The main loop goes through the list of problems and processes them in several steps: (1) similarity search and retrival of relevant information, (2) prompt creation, (3) LLM query, and (4) merging the results of LLM calls into a global list of strategies and data entries to ensure consistency and usability of the output. The following is a detailed description of this four-fold process.

Within the `process_problems` function, we go through a list of problems and process each one as follows: First, we perform a similarity search using the vector store (`vector_store.similarity_search(problem, k=3)`). This retrieves the most relevant document chunks containing context related to the problem. Second, we construct a prompt using `PromptTemplate`, which takes the problem and context as input. The constructed prompt includes both the problem and the relevant information extracted from the documents. In the third step, this prompt is used to invoke the structured language model (`structured_llm_problem.ainvoke(prompt)`). These first three steps demonstrate the concept of retrieval-augmented generation (RAG), where the LLM generates responses based on retrieved context. In our case, the generated response is also formatted to include both conversation strategies and data entries, represented as instances of the `ProblemResponse` model. This approach allows the LLM to produce output in a structured Python format using information stored in our vector database.

In our case, we also have an additional fourth step of merging all the chunks of data. After receiving the response from the LLM, the results of the call to the language model are merged into one final data structure, ensuring that all generated strategies and data entries are consolidated into a cohesive format for further use. This involves adjusting the numbering of the generated strategies by iterating through `assistant_response.strategies`, extracting the local strategy numbers, and assigning them new global numbers using `strategy_counter`. The numbering is adjusted so that the strategies are uniquely and consistently numbered across different problems. This helps maintain an organized list of strategies that can be referenced easily in subsequent interactions.

The data entries (`assistant_response.data`) are also adjusted to reflect the new numbering scheme for strategies. Each data entry contains a sample message (`girlfriend_message`) and an `expected_strategy` field that corresponds to a specific strategy. The `expected_strategy` is updated by mapping the local strategy number to its global counterpart using `strategy_number_mapping`. This ensures that the strategy referenced in each data entry is consistent with the globally numbered strategy list.

Once all problems have been processed, the `AssistantResponse` model is instantiated with the collected strategies and data entries. This final validation step ensures that the number of generated strategies and data entries meets the minimum length constraints, making the output suitable for further use in conversation simulations.

The program outputs two files, for strategies and data entries respectively, and places them in the [output folder](./strategies_from_documents_4/output). The obtained information can then be used for a chatbot or for subsequent development and testing.

## Psychology Lab

**File**: [psychology\_lab\_5/psychology\_lab.py](./psychology_lab_5/psychology_lab.py)

This script demonstrates the use of agent-based simulations for chatbot refinement. The goal is to iteratively refine a 'boyfriend' chatbot's list of strategies using background knowledge og LLM. We will use [REFLECTION pattern](https://langchain-ai.github.io/langgraph/tutorials/reflection/reflection/) from LangGraph tutorial and adapt it to our context. 

In our psychology lab, we have two agents: a psychologist expert and an assistant psychologist. They work together to evaluate and improve the chatbot's conversation strategies by using reflection. Reflection involves prompting an LLM to observe its past actions, assess their quality, and use this evaluation for re-planning or further improvements.

The psychologist agent reviews the chatbot's strategies and data, providing feedback on how to improve them. This feedback includes checking if the strategies are suitable, effective, and aligned with human emotional responses. The assistant agent then applies these suggestions, refining the strategies and data. This iterative process continues, with the psychologist providing feedback and the assistant making improvements, ultimately enhancing the chatbot's empathy and support quality.

The conversation state is tracked using a `PsychologyLabState` object, which records the strategies, data entries, iteration count, and messages exchanged throughout the simulation. The state serves as a central reference for the entire refinement process:

```python
# Define the state class
class PsychologyLabState(TypedDict):
    strategies: List[str]
    data: List[Dict]
    messages: List[BaseMessage]
    iteration: int
```

The psychologist (expert\_node) and assistant (assistant\_node) agents interact with this state to continuously improve the chatbot's strategies. Initially, input data is loaded from JSON files containing conversation strategies and data entries. These files are used to populate the `PsychologyLabState`, including the initial strategies, data, iteration count, and messages. The psychologist expert reviews the current `strategies` and `data` to identify gaps and provide feedback. This feedback is added to the `messages` and used as an input for the task of the assistant psychologist wos job is to update and enhance the content by refining the strategies and adding or adjusting data entries. The assistant outputs the result in a structured format similar to format we used in the previous section. The assistant  saves the updated strategies and data back into the state.

The iterations are controlled by the `should_continue` function, which checks the current iteration count against a maximum allowed value (`max_iterations`). Initially, the workflow starts with the expert reviewing and providing feedback (`expert_node`), followed by the assistant updating the strategies (`assistant_node`). After each iteration, the `should_continue` function determines whether to proceed with another cycle or to move to the final save phase (`save_files`). This iterative process gradually refines the chatbot's conversational strategies, developing a foundation for a more empathetic and authentic chatbot.

The implementation of the program is described in [`psychology_lab.py`](./psychology_lab_5/psychology_lab.py). This script sets up the conversation graph and defines prompts for the psychologist and assistant agents. As the workflow runs through multiple iterations for continuous improvement, intermediate results are saved in the work folder. You can explore the [input folder](./psychology_lab_5/input) for initial data, the [work folder](./psychology_lab_5/work) for intermediate results, and the [output folder](./psychology_lab_5/output) for the final outputs.

## Chatbot Workbench

**Directory**: [chatbot\_workbench\_6](./chatbot_workbench_6)

This final section ties all the previous content together. It shows how to create a comprehensive workbench to iteratively develop and refine chatbots using LangGraph. The Workbench includes two programs.

### Benchmarking Strategies and Prompts

The program [chatbot\_workbench\_6/strategy\_selection\_test.py](./chatbot_workbench_6/strategy_selection_test.py) simulates conversations between a boyfriend and girlfriend, testing different conversation strategies to see how they affect the emotional outcome of the interaction. The BoyfriendNode structure and functions have been described in a   section "Adapting Self-Discover Framework to Modeling Emotional Intelligence". Different method of generating  information stored in data.json and strategies.json files has also been described in the previous sections. The provided code brings everything together to create the test platform. 

The code functions as a benchmarking tool to evaluate the effectiveness of different conversation strategies in simulated dialogues between AI agents acting as a boyfriend and a girlfriend. By running multiple simulations with randomly selected initial messages, the code collects detailed data on each interaction, including mood detection, strategy selection, responses, and the girlfriend's emotional reactions and justifications. This data is saved in a JSON format for analysis.

Through this process, the code enables the assessment of which strategies lead to positive, neutral, or negative emotional outcomes. The collected data allows for the identification of patterns and the evaluation of strategy performance, providing valuable insights for refining conversational strategies and improving AI communication models. Ultimately, the benchmarking facilitates the development of more effective conversational AI by systematically testing and analyzing the impact of different strategies on emotional responses.

### Testing Strategies through Agent-Based Simulation

The program [`strategies_agent_based_simulation.py`](./chatbot_workbench_6/strategies_agent_based_simulation.py) implements a comprehensive agent-based simulation for testing conversation strategies in the context of the chatbot workbench. The program initializes both the boyfriend and girlfriend agents, each having distinct roles defined by prompts and action sequences.

The program interacts with several files to facilitate these simulations, including `strategies.json` for communication strategies, `data.json` for initial messages. The boyfriend agent uses communication strategies from `strategies.json`, while the simulation cycle relies on data entries stored in `data.json`. The boyfriend agent follows the Empathetic Boyfriend pattern used before: it detects the girlfriend's mood, selects an appropriate communication strategy, and generates a response, while the girlfriend agent responds based on her perceived emotional state.

The program captures agent interactions in a structured conversation log, allowing for a detailed analysis of the strategies used and their effects on the emotional dynamics of the conversation. The content of the log is saved in the `output.json` file. By running multiple iterations of these simulated dialogues, the program collects valuable data that can be used to refine and enhance the chatbot's emotional intelligence, ultimately improving its response quality.

I hope this tutorial helps you design, develop, and test your own emotionally intelligent chatbots.
