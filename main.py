from crewai import Agent, Task, Crew
from flask import Flask, jsonify, request
from llms_and_agents import poet_haiku, poet_sonnet

# Create the haiku task with the provided word
write_poetry_task = Task(
    description='Write about {input}.',
    expected_output='Write your poetry. No special characters or newline characters.',
    agent=poet_sonnet
)

# Create and run the crew
crew = Crew(
    agents=[poet_sonnet],
    tasks=[write_poetry_task],
    verbose=False
)

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def call_agent():
    try:
        input = request.args.get('input', '???')
        
        result = crew.kickoff(inputs={'input': input})
        result_str = str(result)
        return jsonify({
            'input': input,
            'raw': result_str
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 