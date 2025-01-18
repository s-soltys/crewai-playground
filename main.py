from flask import Flask, jsonify, request
from llms_and_agents import crew


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