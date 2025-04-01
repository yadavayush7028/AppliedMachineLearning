from flask import Flask, request, jsonify, render_template
import joblib
from score import score
import os

app = Flask(__name__)

# Load the trained model
# You might want to update the path based on your actual model location
MODEL_PATH = os.environ.get('MODEL_PATH', 'model.pkl')
model = joblib.load(MODEL_PATH)

# Default threshold
DEFAULT_THRESHOLD = 0.5

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score_endpoint():
    """Endpoint to score text for spam classification"""
    # Check if request contains JSON
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    # Get text from request
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    text = data['text']
    
    # Get threshold from request or use default
    threshold = data.get('threshold', DEFAULT_THRESHOLD)
    
    try:
        # Score the text
        prediction, propensity = score(text, model, threshold)
        
        # Return results as JSON
        return jsonify({
            "prediction": int(prediction),  # Convert boolean to int (0 or 1)
            "propensity": propensity
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)