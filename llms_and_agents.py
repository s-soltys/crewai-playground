from crewai import Agent
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


# Create a poet agent
poet_haiku = Agent(
    role='Haiku Poet',
    goal='Write beautiful and meaningful haikus',
    backstory='An experienced poet with deep knowledge of both haiku',
    llm=openai_llm,
    verbose=False
)

poet_sonnet = Agent(
    role='Sonnet Poet',
    goal='Write beautiful and meaningful sonnets',
    backstory='An experienced poet with deep knowledge of both sonnet',
    llm=openai_llm,
    verbose=False
)