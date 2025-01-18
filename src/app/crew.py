from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from datetime import datetime
from crewai_tools import (SerperDevTool, DallETool)

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class App():
	"""App crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def news_researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['news_researcher'],
			tools=[SerperDevTool()]
		)

	@agent
	def summary_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['summary_writer'],
		)

	@agent
	def poet(self) -> Agent:
		return Agent(
			config=self.agents_config['poet'],
		)

	@agent
	def painter(self) -> Agent:
		return Agent(
			config=self.agents_config['painter'],
			tools=[DallETool()]
		)
	
	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_news_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_news_task'],
		)

	@task
	def summarise_task(self) -> Task:
		return Task(
			config=self.tasks_config['summarise_task'],
			output_file=f'output/report_{datetime.now().strftime("%Y-%m-%d_%I-%M-%p")}.md'
		)

	@task
	def write_poem_task(self) -> Task:
		return Task(
			config=self.tasks_config['write_poem_task'],
			output_file=f'output/poem_{datetime.now().strftime("%Y-%m-%d_%I-%M-%p")}.md'
		)

	@task
	def paint_task(self) -> Task:
		return Task(
			config=self.tasks_config['paint_task'],
			output_file=f'output/image_{datetime.now().strftime("%Y-%m-%d_%I-%M-%p")}.png'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the App crew"""
		return Crew(
			agents=self.agents,
			tasks=[
				self.research_news_task(),
				self.summarise_task(),
				self.write_poem_task()
			],
			process=Process.sequential,
			verbose=True,
		)
