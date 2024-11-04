# strategies_from_documents.py
# Copyright (c) 2024 skrivov
import os
import json
import asyncio
from typing import List, Dict, Set
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LangChain components
# from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate

# Ensure OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set your OpenAI API key in the OPENAI_API_KEY environment variable.")

# Define Pydantic models for structured output
class Strategy(BaseModel):
    title: str = Field(description="Short title of the strategy, composed of 2 to 4 words")
    description: str = Field(description="Description of the strategy")

class DataEntry(BaseModel):
    girlfriend_message: str = Field(description="Sample message from the girlfriend")
    expected_strategy: str = Field(description="Number of the expected strategy as a string")

class ProblemResponse(BaseModel):
    strategies: List[Strategy] = Field(
        description="List of communication strategies, each with a title and description."
    )
    data: List[DataEntry] = Field(
        description="List of data entries with 'girlfriend_message' and 'expected_strategy'"
    )

class AssistantResponse(BaseModel):
    strategies: List[Strategy] = Field(
        description="List of at least 20 unique conversation strategies, each with a title and description.",
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

# Define the list of problems/opportunities
problems = [
    "She is feeling stressed about work.",
    "She is upset about a disagreement with a friend.",
    "She feels unappreciated.",
    "She is bored and looking for something fun to do.",
    "She is anxious about an upcoming exam.",
    "She feels lonely because you haven't spent much time together.",
    "She is sad about a family issue.",
    "She is frustrated with a project that isn't going well.",
    "She wants to celebrate a recent achievement.",
    "She is in a bad mood for no apparent reason.",
    "She is excited about planning a trip.",
    "She feels overwhelmed with responsibilities.",
    "She is disappointed about canceled plans.",
    "She is curious about trying a new hobby.",
    "She is feeling nostalgic about old memories.",
    "She is worried about her health.",
    "She wants to discuss future plans together.",
    "She is happy and wants to share her joy.",
    "She is feeling insecure about the relationship.",
    "She needs advice on a difficult decision."
]

# List of URLs to load documents from
urls = [
    "https://29k.org/article/how-to-show-empathy-in-communication-statement-examples",
    "https://www.marriage.com/advice/relationship/how-to-make-your-girlfriend-happy/",
    "https://www.stylecraze.com/articles/how-to-make-girlfriend-happy/"
]

# Async function to load documents wuth Unstructured Loader
# async def load_documents(urls: List[str]) -> List[Document]:
#     docs = []
#     for url in urls:
#         loader = UnstructuredLoader(web_url=url)
#         try:
#             async for doc in loader.alazy_load():
#                 docs.append(doc)
#         except Exception as e:
#             print(f"Error loading {url}: {e}")
#     return docs

# Async function to load documents with WebBasedLoader
async def load_documents(urls: List[str]) -> List[Document]:
    docs = []
    for url in urls:
        loader = WebBaseLoader(
            web_paths=[url],
            # Customize bs_kwargs and bs_get_text_kwargs if necessary
        )
        try:
            async for doc in loader.alazy_load():
                docs.append(doc)
        except Exception as e:
            print(f"Error loading {url}: {e}")
    return docs

# Load documents with WebBasedLoader. Comment out if using Unstructured Loader
docs = asyncio.run(load_documents(urls))

# Load documents with Unstructured Loader
# split_docs = asyncio.run(load_documents(urls))

# Check if any documents were loaded
if not docs:
    raise ValueError("No documents were loaded. Please check the URLs and try again.")

# Comment out code for spliting docs if using Unstructured Loader
# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # Adjust the chunk size as needed
    chunk_overlap=100  # Overlap between chunks to maintain context
)

# Split documents into chunks
split_docs = []
for doc in docs:
    splits = text_splitter.split_text(doc.page_content)
    for i, chunk in enumerate(splits):
        new_doc = Document(
            page_content=chunk,
            metadata=doc.metadata  # Retain the metadata
        )
        split_docs.append(new_doc)

print("Splits:   ", len(split_docs))
# End comment out section

# Create InMemoryVectorStore
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore.from_documents(split_docs, embeddings)

# Function to process problems
async def process_problems(problems: List[str], vector_store, structured_llm_problem):
    all_strategies: List[Strategy] = []
    all_data_entries: List[DataEntry] = []
    existing_titles: Set[str] = set()
    title_to_global_number: Dict[str, int] = {}
    strategy_counter = 1  # Global strategy number

    for idx, problem in enumerate(problems):
        # Perform similarity search
        retrieved_docs = vector_store.similarity_search(problem, k=3)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Prepare the prompt
        prompt_template = PromptTemplate(
            input_variables=["problem", "context"],
            template="""
You are a relationship expert.

Given the following problem:

"{problem}"

And the following context extracted from articles:

"{context}"

Using the context, generate 1 to 3 communication strategies to address this problem, suitable for a boyfriend chatbot communicating via text messages.

Each strategy should include a 'title' and 'description' as separate fields.

Also, provide 2 to 3 example messages from the girlfriend that would match this problem, along with the expected strategy number (corresponding to the position of the strategy in the strategies list, starting from 1).

Provide the updated strategies and data in JSON format matching the schema of ProblemResponse.

Remember that the strategies should be suitable for text-based communication and that non-verbal cues are not possible.
"""
        )

        prompt = prompt_template.format(
            problem=problem,
            context=context
        )

        print("Processing problem:", problem)
        # print("Context:", context)  # Uncomment if you want to see the context

        try:
            # Generate response from the structured LLM for ProblemResponse
            assistant_response = await structured_llm_problem.ainvoke(prompt)
            # assistant_response is an instance of ProblemResponse
        except ValidationError as e:
            print(f"Validation Error in assistant response for problem '{problem}': {e}")
            continue
        except Exception as e:
            print(f"Error processing problem '{problem}': {e}")
            continue

        # Mapping from local index to global strategy number
        local_to_global_number: Dict[int, int] = {}

        for local_index, strategy in enumerate(assistant_response.strategies, start=1):
            title = strategy.title.strip()
            description = strategy.description.strip()

            if title not in existing_titles:
                # Strategy is new
                existing_titles.add(title)
                all_strategies.append(Strategy(title=title, description=description))
                title_to_global_number[title] = strategy_counter
                local_to_global_number[local_index] = strategy_counter
                strategy_counter += 1
            else:
                # Strategy title already exists
                # We do not add it to all_strategies again
                # Map local index to existing global number
                local_to_global_number[local_index] = title_to_global_number[title]

        # Adjust expected_strategy in data entries
        for data_entry in assistant_response.data:
            try:
                local_expected_strategy = int(data_entry.expected_strategy.strip())
            except ValueError:
                print(f"Invalid expected_strategy '{data_entry.expected_strategy}' in data entry. Skipping.")
                continue

            if local_expected_strategy in local_to_global_number:
                global_expected_strategy = local_to_global_number[local_expected_strategy]
                data_entry.expected_strategy = str(global_expected_strategy)
                all_data_entries.append(data_entry)
            else:
                # The strategy was not added due to duplication; skip this data entry
                print(f"Data entry refers to a strategy that was not added. Skipping data entry.")
                continue

    return all_strategies, all_data_entries

# Process problems
all_strategies, all_data_entries = asyncio.run(process_problems(problems, vector_store, structured_llm_problem))

# Ensure minimum length constraints
try:
    assistant_response = AssistantResponse(
        strategies=all_strategies,
        data=all_data_entries
    )
except ValidationError as e:
    print(f"Validation error: {e}")
    # Handle the error or adjust data accordingly
    # For simplicity, we'll stop execution here
    exit(1)

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)

# Save strategies
strategies_path = os.path.join(output_dir, 'strategies.json')
formatted_strategies = []
for idx, strategy in enumerate(assistant_response.strategies, start=1):
    formatted_strategy = f"{idx}. **{strategy.title}**: {strategy.description}"
    formatted_strategies.append(formatted_strategy)

with open(strategies_path, 'w') as f:
    json.dump({'conversation_strategies': formatted_strategies}, f, indent=2)

# Save data
data_path = os.path.join(output_dir, 'data.json')
data_entries = [entry.model_dump() for entry in assistant_response.data]
with open(data_path, 'w') as f:
    json.dump(data_entries, f, indent=2)

print("Strategies and data have been saved to the 'output' folder.")
