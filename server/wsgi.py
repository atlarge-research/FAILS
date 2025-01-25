from app import app
from scripts.analysis import analyze_plot, summarize_analyses
from flask import request, jsonify
import matplotlib.pyplot as plt
import base64
import io

if __name__ == "__main__":
    app.run() 

@app.route('/api/analyze-plot', methods=['POST'])
def analyze_plot_endpoint():
    try:
        # Get the image data from the request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
            
        image_base64 = data['image']
        
        # Verify base64 string is not empty
        if not image_base64:
            return jsonify({"error": "Empty image data"}), 400
            
        print(f"Image size: {len(image_base64)} bytes")  # Debug logging
        
        # Analyze the image
        result = analyze_plot(image_base64)
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify({"error": result["error"]}), 500
            
    except Exception as e:
        print(f"Server error in analyze_plot_endpoint: {str(e)}")  # Logging
        return jsonify({"error": str(e)}), 500 

@app.route('/api/summarize-analyses', methods=['POST'])
def summarize_analyses_endpoint():
    try:
        data = request.get_json()
        if not data or 'analyses' not in data:
            return jsonify({"error": "No analyses provided"}), 400
            
        result = summarize_analyses(data['analyses'])
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify({"error": result["error"]}), 500
            
    except Exception as e:
        print(f"Server error in summarize_analyses_endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500 