from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch
import json

app = Flask(__name__)

# Load fine-tuned model
model = SentenceTransformer('./fine-tuned-miniLM')

# Load your QA data
with open("output.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)["qa_pairs"]

answers = [item["Answer"] for item in qa_pairs]
questions = [item["Question"] for item in qa_pairs]

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Embed the query
    query_embedding = model.encode(question, convert_to_tensor=True)

    # ⬇️ Encode answers here instead of in memory at startup
    answer_embeddings = model.encode(answers, convert_to_tensor=True)

    # Search top matches using cosine similarity
    hits = util.semantic_search(query_embedding, answer_embeddings, top_k=3)[0]

    results = []
    for hit in hits:
        results.append({
            "score": float(hit['score']),
            "answer": answers[hit['corpus_id']],
            "matched_question": questions[hit['corpus_id']]
        })

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(port=5000)
