from flask import Flask, jsonify, request, redirect, url_for
import os
import json
import requests
from flask_swagger_ui import get_swaggerui_blueprint
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv() 

# Define the models file URL or local file path as global variables
models_file_url = os.environ.get('MODELS_FILE_URL')
models_file_path = os.environ.get('MODELS_FILE_PATH')

def fetch_models():
    if models_file_url:
        response = requests.get(models_file_url)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    elif models_file_path:
        try:
            with open(models_file_path, 'r',encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error reading local file: {e}")
            return None
    else:
        print("No source for models file specified")
        return None

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('swagger_ui.show'))

@app.route('/models', methods=['GET'])
def get_models():
    models = fetch_models()
    if models is not None:
        return jsonify(models)
    else:
        return jsonify({"error": "Unable to fetch models from URL or local file"}), 404

@app.route('/model', methods=['GET'])
def get_model():
    model_id = request.args.get('id')
    model_name = request.args.get('name')
    models = fetch_models()
    if models is not None:
        if model_id or model_name:
            for model in models:
                if model_id and model['id'] == int(model_id):
                    return jsonify(model)
                if model_name and model['name'].lower() == model_name.lower():
                    return jsonify(model)
            return jsonify({"error": "Model not found"}), 404
        else:
            return jsonify({"error": "Model ID or Name required i.e ?id=1 or ?name=gpt-4o"}), 400
    else:
        return jsonify({"error": "Unable to fetch models from URL or local file"}), 404

@app.route('/count', methods=['GET'])
def count_models():
    models = fetch_models()
    if models is not None:
        return jsonify({"count": len(models)})
    else:
        return jsonify({"error": "Unable to fetch models from URL or local file"}), 404

@app.route('/search', methods=['GET'])
def search_models_by_name():
    search_term = request.args.get('name')
    models = fetch_models()
    if models is not None:
        matching_models = [model for model in models if search_term.lower() in model['name'].lower()]
        return jsonify(matching_models)
    else:
        return jsonify({"error": "Unable to fetch models from URL or local file"}), 404

# Swagger UI setup
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.yaml'  # Path to your OpenAPI file

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Foundry API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    app.run(debug=True)