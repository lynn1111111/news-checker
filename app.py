from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# ✅ 경량 모델로 교체
classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1"
)

# 정치 성향 라벨
labels = ["보수", "중도", "진보"]

# ✅ 모델에 전달할 최대 입력 길이 설정 (문자 수 기준)
MAX_MODEL_TEXT_LEN = 1000

# 근거 문장 추출 함수
def find_reason_sentence(text, predicted_label):
    sentences = text.split('다.')
    best_sentence = None
    best_score = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        try:
            result = classifier(sentence + "다.", candidate_labels=labels)
            for label, score in zip(result['labels'], result['scores']):
                if label == predicted_label and score > best_score:
                    best_score = score
                    best_sentence = sentence + "다."
        except Exception:
            continue  # 예외 발생 시 무시하고 다음 문장 분석

    return best_sentence if best_sentence else "모델이 판단한 전체 본문 기준 결과입니다."

# 홈 화면
@app.route('/')
def index():
    return render_template('index.html')

# 분석 요청 처리
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['article']
    if not text.strip():
        return render_template('result.html', final_label="분석할 내용이 없습니다.", confidence=0, reason_sentence=None)

    # ✅ 모델 입력 길이 제한 적용
    model_input = text[:MAX_MODEL_TEXT_LEN]

    result = classifier(model_input, candidate_labels=labels)
    predicted_label = result['labels'][0]
    confidence = result['scores'][0]

    reason_sentence = find_reason_sentence(text, predicted_label)

    return render_template('result.html',
                           final_label=predicted_label + (f" (△ 신뢰도 낮음)" if confidence < 0.5 else ""),
                           confidence=round(confidence * 100, 2),
                           reason_sentence=reason_sentence)

if __name__ == '__main__':
    app.run()
