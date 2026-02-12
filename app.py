import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import JobRecommender

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

recommender = None


def initialize_app():
    global recommender
    try:
        logger.info("Initializing Job Recommender...")
        recommender = JobRecommender(artifacts_dir="artifacts")
        logger.info("Recommender initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize recommender.")
        recommender = None


@app.route("/recommend", methods=["POST"])
def get_recommendations():
    if recommender is None:
        return jsonify({"error": "Recommender system not initialized"}), 500

    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field in request body"}), 400

    query = data.get("query")
    top_k = data.get("top_k", 10)

    # Validate query
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Query must be a non-empty string"}), 400

    # Validate top_k
    if not isinstance(top_k, int) or top_k <= 0:
        return jsonify({"error": "top_k must be a positive integer"}), 400

    # Optional safety cap for API performance
    top_k = min(top_k, 50)

    try:
        results = recommender.recommend(query, top_k=top_k)
        return jsonify(results), 200
    except Exception:
        logger.exception("Error during inference.")
        return jsonify(
            {"error": "Internal server error during recommendation"}
        ), 500


@app.route("/health", methods=["GET"])
def health_check():
    status = "ready" if recommender is not None else "initializing/error"
    return jsonify({"status": status}), 200


if __name__ == "__main__":
    initialize_app()
    app.run(host="0.0.0.0", port=5000, debug=False)