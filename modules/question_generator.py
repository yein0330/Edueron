from typing import List
from src.inference.llm_model import local_llm


INSTRUCTION = """
Given the following list of concept summaries, generate 3 high-quality, relevant practice questions for each concept.
- Questions should be clear and specific.
- Use various formats such as multiple choice, short answer, or fill in the blanks.
- Avoid repeating similar questions.

Format your output clearly under each concept title.

Example:
### Concept: Newton's First Law
1. What does Newton's First Law state about objects in motion?
2. Fill in the blank: An object at rest stays at rest unless acted on by a ________.
3. A ball rolling on a frictionless surface will continue to roll. Which law explains this?
"""


def generate_questions_from_concepts(concept_list: List[str],subject: str) -> str:
    """
    Given a list of concept summaries and subject, return generated questions **with answers**.
    """
    prompt = (
        f"You are a helpful teacher specialized in '{subject}'. "
        f"Based on each concept below, create one or two short-answer practice questions. "
        f"For each question, also provide a clear and accurate answer.\n\n"
    )

    for concept in concept_list:
        prompt += f"### Concept: {concept}\n"

    result = local_llm.generate(prompt)

    # Post-processing (optional, depending on model's style)
    if "### Concept:" in result:
        result = result.split("### Concept:", 1)[-1]
        result = "### Concept:" + result
    return result.strip()

def parse_generated_questions(text: str) -> list[dict]:
    """
    Parse the generated Q&A text into a list of {"question": ..., "answer": ...} dicts.
    """
    import re
    qa_pairs = []
    current_q, current_a = None, None
    for line in text.strip().splitlines():
        if line.startswith("Q"):
            if current_q and current_a:
                qa_pairs.append({"question": current_q.strip(), "answer": current_a.strip()})
                current_q, current_a = None, None
            current_q = re.sub(r'^Q\d*\.\s*', '', line)
        elif line.startswith("A"):
            current_a = re.sub(r'^A\d*\.\s*', '', line)

    # 마지막 QA도 추가
    if current_q and current_a:
        qa_pairs.append({"question": current_q.strip(), "answer": current_a.strip()})

    return qa_pairs


if __name__ == "__main__":
    sample_concepts = [
        "Photosynthesis and light-dependent reactions",
        "Newton's Laws of Motion",
        "Pythagorean theorem in right-angled triangles"
    ]
    questions = generate_questions_from_concepts(sample_concepts)
    print(questions)