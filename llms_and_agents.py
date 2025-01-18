from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize LLMs
openai_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

ollama_llm = ChatOpenAI(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)

gemma_llm = ChatOpenAI(
    model="ollama/gemma:2b",
    base_url="http://localhost:11434"
)

lmstudio_llm = ChatOpenAI(
    model="mlx-community/llama-3.2-3b-instruct",
    base_url="http://localhost:1234"
)

# Initialize search tool
search = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))
search_tool = Tool(
    name="Search",
    func=search.run,
    description="Search the internet for information about poems and poetry"
)


# Create a poet agent
manager_poet = Agent(
    role='Manager Poet',
    goal='Asks other poets to provide poems',
    backstory='A poet manager',
    llm=openai_llm,
    verbose=True
)

# Create a poet agent
ai_poet = Agent(
    role='Local Poet',
    goal='Write beautiful and meaningful poem',
    backstory='An experienced poet with deep knowledge of both haiku',
    llm=ollama_llm,
    verbose=True
)

real_poet = Agent(
    role='Poet Librarian',
    goal='Find beautiful and meaningful sonnets',
    backstory='An experienced poet librarian with deep knowledge of many poems who can find them on the internet',
    llm=ollama_llm,
    tools=[search_tool],
    verbose=True
)

# Create the haiku task with the provided word
write_poetry_task = Task(
    description='Write poem about {input}.',
    expected_output='Title of the poem, and the poem found about {input}.',
    agent=ai_poet
)

# Create the haiku task with the provided word
find_poem_task = Task(
    description='Find poem about {input}.',
    expected_output='Title of the poem, and the poem found about {input}.',
    agent=real_poet
)

pick_best_poem_task = Task(
    description='Pick the best poem out of two poems by different angents',
    expected_output='Two poems listed one by one, and the best one + reasons why it is the best',
    agent=manager_poet
)

# Create and run the crew
crew = Crew(
    agents=[ai_poet, real_poet],
    tasks=[find_poem_task, write_poetry_task, pick_best_poem_task],  # Tasks to be delegated and executed under the manager's supervision
    process=Process.hierarchical,  # Specifies the hierarchical management approach
    respect_context_window=True,  # Enable respect of the context window for tasks
    memory=True,  # Enable memory usage for enhanced task execution
    manager_agent=manager_poet,  # Optional: explicitly set a specific agent as manager instead of the manager_llm
    planning=True,  # Enable planning feature for pre-execution strategy
    verbose=True
)