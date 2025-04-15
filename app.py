from flask import Flask, render_template, request
from transformers import pipeline
import torch

app = Flask(__name__)

# ✅ 모델 및 토크나이저 설정 (제로샷 분류 파이프라인)
classifier = pipeline(
    "zero-shot-classification",  # ✔ 필수 수정
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

# ✅ 정치 성향 라벨 정의
labels = ["보수", "중도", "진보"]

# ✅ 문장에서 근거 문장 추출 함수
def find_reason_sentence(text, predicted_label):
    sentences = text.split('다.')  # 문장 분리 (개선 여지 있음)
    best_sentence = None
    best_score = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # 문장에 '다.'를 다시 붙여줌
        result = classifier(sentence + "다.", candidate_labels=labels)
        for res in result['scores']:
            if result['labels'][result['scores'].index(res)] == predicted_label and res > best_score:
                best_score = res
                best_sentence = sentence + "다."

    return best_sentence if best_sentence else "모델이 판단한 전체 본문 기준 결과입니다."

# ✅ 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# ✅ 분석 처리
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['article']
    if not text.strip():
        return render_template('result.html', final_label="분석할 내용이 없습니다.", confidence=0, reason_sentence=None)

    # 정치 성향 예측
    result = classifier(text, candidate_labels=labels)
    scores = result['scores']
    labels_sorted = result['labels']
    top_result_idx = scores.index(max(scores))
    predicted_label = labels_sorted[top_result_idx]
    confidence = scores[top_result_idx]

    # 근거 문장 추출
    reason_sentence = find_reason_sentence(text, predicted_label)

    return render_template('result.html',
                           final_label=predicted_label + (f" (△ 신뢰도 낮음)" if confidence < 0.5 else ""),
                           confidence=round(confidence * 100, 2),
                           reason_sentence=reason_sentence)

if __name__ == '__main__':
    app.run(debug=True)
