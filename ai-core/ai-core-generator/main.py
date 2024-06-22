from flask import Flask, request, jsonify
from ShakespeareanGenerator.generator import Generator, ModelManager
from ShakespeareanGenerator.logger import Logger

app = Flask(__name__)
app.json.sort_keys = False

model_manager = ModelManager()
logging = Logger()

def load_model():
    try:
        if not model_manager.is_model_loaded():
            model_manager.load_model()
        else:
            logging.info("Model already loaded")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

@app.before_request
def initialize():
    load_model()

@app.route('/v2/generate', methods=["POST"])
def generate_text():
    data = request.get_json()
    max_tokens = int(data.get('max_tokens', 300))
    temperature = float(data.get('temperature', 1.0))
    top_k = int(data.get('top_k', 0))
    top_p = float(data.get('top_p', 0.9))

    generator = Generator(model_manager, max_tokens, temperature, top_k=top_k, top_p=top_p)
    generated_text = generator.generate()
    processed_text = generator.post_process_text(generated_text)
    lines = [line.strip() for line in processed_text.split('.') if line.strip()]

    response = {
        'generated_text': lines,
        'model_details': {
            'model_name': 'shakespeare-language-model',
            'temperature': generator.temperature, 
            'length': generator.length,
            'top_k': generator.top_k,
            'top_p': generator.top_p,
        }
    }
    return jsonify(response)

if __name__ == '__main__':
    app.debug = False
    app.run(host='0.0.0.0', port=9001)
