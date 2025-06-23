# src/modules/concept_summary.py

from typing import List
from src.inference.llm_model import local_llm

def extract_concepts_as_list(summary_text: str) -> List[str]:
    """
    Extract each line of the concept summary as a clean concept string list.
    """
    return [
        line.strip("•").strip("-").strip()
        for line in summary_text.splitlines()
        if line.strip()
    ]

def summarize_concepts_from_questions(pdf_input:str, subject: str) -> str:
    """
    Extract raw text from a PDF file and summarize its key concepts using LLM.

    Parameters
    ----------
    pdf_input : str
        PDF file converted into text.
    subject : str
        Subject name (e.g., "수학", "영어", etc.)

    Returns
    -------
    List[str]
        List of summarized key concepts.
    """

    prompt = (
            f"You're an experienced teacher helping middle and high school students study for the subject '{subject}'.\n"
            f"The following text contains a list of exam questions extracted from a real test paper.\n"
            f"Please carefully analyze **each question individually**, and extract the core concept or skill it is testing.\n\n"
            "Instructions:\n"
            "- For each question, write a single bullet point summarizing the concept it covers.\n"
            "- Use clear and simple language that students can understand.\n"
            "- Do not include answers, only the concept or skill being tested.\n"
            "- If multiple questions test the same concept, write it again for clarity.\n"
            "- If a question is unclear or incomplete, write '🔸 내용 불충분함'.\n\n"
            "Exam Questions:\n"
            f"{pdf_input}\n\n"
            "Concept Summary:"
            )


    response = local_llm.generate(prompt)
    return extract_concepts_as_list(response)