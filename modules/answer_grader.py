# src/modules/answer_grader.py
from difflib import SequenceMatcher

from src.inference.llm_model import local_llm
from src.inference.embedding_model import embedding_model
from src.utils.similarity import cosine_similarity

def sequence_similarity_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()

def vector_similarity_score(a: str, b: str) -> float:
    vec1 = embedding_model.get_embedding(a)
    vec2 = embedding_model.get_embedding(b)
    return cosine_similarity(vec1, vec2)

def simple_text_similarity(a: str, b: str) -> float:
    """
    Compute a similarity score between two strings using SequenceMatcher.
    """
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()

def grade_answer(student_text: str, correct_answer: str, subject: str,question: str = "") -> str:
    student_text = student_text.strip()
    correct_answer = correct_answer.strip()
    question = question.strip()
    
    if not student_text:
        return "✘ 답안이 인식되지 않았습니다. 다시 입력하거나 업로드해주세요."

    if subject in ["국어", "영어"]:
        # 의미 유사도 (임베딩 기반)
        score = vector_similarity_score(student_text, correct_answer)
        score_pct = int(score * 100)
        feedback = f"의미 기반 유사도 점수는 {score_pct}점입니다."
    
    elif subject == "수학":
        # 문자열 유사도 (문자 일치 중심)
        score = sequence_similarity_score(student_text, correct_answer)
        score_pct = int(score * 100)
        feedback = f"문자 기반 유사도 점수는 {score_pct}점입니다."
    else:
        # 기타 과목은 LLM 기반 채점
        prompt = f"""
        [과목: {subject}]
        정답: "{correct_answer}"
        학생 답안: "{student_text}"
        위 학생의 답안을 채점하고, 점수(100점 만점 기준)와 간단한 피드백을 제공해주세요.
        """.strip()
        try:
            response = local_llm.generate(prompt)
            feedback = response.split("=== Answer ===")[-1].strip() if "=== Answer ===" in response else response
        except Exception:
            feedback = "⚠️ 채점 중 오류가 발생했습니다. 다시 시도해주세요."
    
    return feedback
