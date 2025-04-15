from flask import Flask, request, render_template
import requests
import os

app = Flask(__name__)

# Hugging Face API 설정
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

LABELS = ["보수", "중도", "진보"]

def query_huggingface_api(text):
    response = requests.post(API_URL, headers=HEADERS, json={
        "inputs": text,
        "parameters": {"candidate_labels": LABELS},
    })
    return response.json()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        news_text = request.form["news"]
        output = query_huggingface_api(news_text)
        if "labels" in output and "scores" in output:
            label = output["labels"][0]
            score = round(output["scores"][0] * 100, 2)
            result = f"{label} ({score}%)"
        else:
            result = "분석 실패: Hugging Face 응답 오류"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
