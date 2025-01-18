from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
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


poet_picker = Agent(
    role='Artist Agent',
    goal='Find the best artist for the job',
    backstory='An experienced artist agent with deep knowledge of the art world',
    llm=openai_llm,
    verbose=True
)

# Create a poet agent
poet_haiku = Agent(
    role='Haiku Poet',
    goal='Write beautiful and meaningful haikus',
    backstory='An experienced poet with deep knowledge of both haiku',
    llm=openai_llm,
    verbose=True
)

poet_sonnet = Agent(
    role='Sonnet Poet',
    goal='Write beautiful and meaningful sonnets',
    backstory='An experienced poet with deep knowledge of both sonnet',
    llm=openai_llm,
    verbose=True
)

# Create the haiku task with the provided word
pick_poet = Task(
    description='Pick the best poet to write about {input}.',
    expected_output='Pick your poet and delegate to them.',
    agent=poet_sonnet
)

# Create the haiku task with the provided word
write_poetry_task = Task(
    description='Write about {input}.',
    expected_output='Write your poetry. No special characters or newline characters.',
    agent=poet_sonnet
)

# Create and run the crew
crew = Crew(
    tasks=[write_poetry_task],  # Tasks to be delegated and executed under the manager's supervision
    agents=[poet_haiku, poet_sonnet],
    manager_llm=openai_llm,  # Mandatory if manager_agent is not set
    process=Process.hierarchical,  # Specifies the hierarchical management approach
    respect_context_window=True,  # Enable respect of the context window for tasks
    memory=True,  # Enable memory usage for enhanced task execution
    manager_agent=None,  # Optional: explicitly set a specific agent as manager instead of the manager_llm
    planning=True,  # Enable planning feature for pre-execution strategy
    verbose=True
)