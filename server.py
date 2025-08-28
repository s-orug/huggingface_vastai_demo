from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        user_text = data.get("text", "")

        if not user_text:
            return jsonify({"error": "Missing 'text' field"}), 400

        messages = [
            {"role": "user", "content": user_text}
        ]

        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model.generate(inputs, max_new_tokens=300, do_sample=True)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
