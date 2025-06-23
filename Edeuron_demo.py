# Edeuron_demo.py

import os
import shutil
from typing import Tuple
import json
import threading
import concurrent.futures
import pickle

import gradio as gr

from src.Edeuron_chatbot import PDFChatBot
from src.utils.auth import hash_password, verify_password, migrate_plain_passwords
from src.scripts import pdf_extractor, chunker, build_index, section_rep_builder
from src.modules.concept_summary import summarize_concepts_from_questions
from src.modules.question_generator import generate_questions_from_concepts, parse_generated_questions
from src.modules.answer_grader import grade_answer
from src.utils.ocr import extract_text_from_image

# ---------------------------------------------------------------------
# Persistent user database (credentials + uploads + prompts)
# ---------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
USER_DB_PATH = os.path.join("data", "user_db.json")

# Re‑entrant lock to guard all reads/writes to the shared user DB in multi‑threaded Gradio
_DB_LOCK = threading.RLock()


def _load_user_db():
    if os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # default structure
    return {"users": {}}


def _save_user_db(db: dict):
    """Persist the in‑memory DB atomically."""
    with _DB_LOCK:
        with open(USER_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2)

DEFAULT_PROMPT = (
    "Answer the user's question based on the information provided in the document context below.\n"
    "Your response should reference the context clearly, but you may paraphrase or summarize appropriately."
)

# Max time (seconds) allowed for pdf_extractor.extract_pdf_content
EXTRACT_TIMEOUT = 120  # 2 minutes

# In‑memory view of the persistent database
_USER_DB = _load_user_db()
# Migrate plain passwords to hashed passwords if needed
_USER_DB = migrate_plain_passwords(_USER_DB)
_save_user_db(_USER_DB)
USERS = {u: info["password"] for u, info in _USER_DB["users"].items()}


def authenticate(username: str, password: str) -> bool:
    """Check if the provided credentials are valid."""
    stored_hash = USERS.get(username)
    if stored_hash is None:
        return False
    # Support both plain and hashed passwords for backward compatibility
    if stored_hash.startswith("$2b$"):
        return verify_password(password, stored_hash)
    else:
        # Legacy plain password comparison (will be migrated on next save)
        return stored_hash == password


def ensure_user_dir(username: str) -> str:
    """Create and return the directory for a specific user."""
    user_dir = os.path.join("data", "user_uploads", username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def login(username: str, password: str):
    if authenticate(username, password):
        return True, username, "Login successful."
    if username not in USERS:
        # Atomically create new user with hashed password
        with _DB_LOCK:
            hashed_pwd = hash_password(password)
            USERS[username] = hashed_pwd
            _USER_DB["users"][username] = {
                "password": hashed_pwd,
                "uploads": [],
                "prompts": [],
            }
            _save_user_db(_USER_DB)
        return True, username, "New user created and logged in."
    return False, "", "Invalid credentials."


def login_and_prepare(username: str, password: str):
    """
    Wrapper for the login flow that also controls component visibility
    and restores the last system prompt after a successful login.
    Returns:
        - logged‑in state (bool)
        - username (str)
        - login message (str)
        - update for the main interaction area (gr.update)
        - update for the prompt textbox (gr.update)
        - update for the existing‑PDF dropdown (gr.update)
    """
    success, uid, msg = login(username, password)

    # Toggle the main area
    main_area_update = gr.update(visible=success)

    # Restore the user's last prompt if available
    prompt_val = DEFAULT_PROMPT
    if success:
        prompts = _USER_DB["users"].get(uid, {}).get("prompts", [])
        if prompts:
            prompt_val = prompts[-1]
    prompt_update = gr.update(value=prompt_val)

    # Populate dropdown with user's previous uploads
    uploads = _USER_DB["users"].get(uid, {}).get("uploads", []) if success else []
    dropdown_update = gr.update(choices=[os.path.basename(u) for u in uploads],
                                value=(os.path.basename(uploads[0]) if uploads else None))

    return success, uid, msg, main_area_update, prompt_update, dropdown_update



# ---------------------------------------------------------------------
# Extraction cache helpers (per‑user, per‑PDF)
# ---------------------------------------------------------------------
def _cache_paths(user_dir: str, pdf_basename: str):
    """Return tuple (sections_path, index_path) inside the user directory."""
    sec_path = os.path.join(user_dir, f"{pdf_basename}_sections.json")
    idx_path = os.path.join(user_dir, f"{pdf_basename}_index.pkl")
    return sec_path, idx_path

def _save_cache(user_dir: str, pdf_basename: str,
                sections: list, chunk_index: list):
    sec_path, idx_path = _cache_paths(user_dir, pdf_basename)
    with open(sec_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    with open(idx_path, "wb") as f:
        pickle.dump(chunk_index, f)

def _load_cache(user_dir: str, pdf_basename: str):
    sec_path, idx_path = _cache_paths(user_dir, pdf_basename)
    if os.path.exists(sec_path) and os.path.exists(idx_path):
        try:
            with open(sec_path, "r", encoding="utf-8") as f:
                sections = json.load(f)
            with open(idx_path, "rb") as f:
                chunk_index = pickle.load(f)
            return sections, chunk_index
        except Exception:
            # corrupted cache – ignore
            pass
    return None, None


def process_pdf(pdf_path: str, user_dir: str, timeout: int = EXTRACT_TIMEOUT) -> Tuple[list, list]:
    """
    Run the extraction/index pipeline with a timeout guard.
    Results are cached to disk inside user_dir for later reuse.
    """
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

    # Run extractor with timeout
    def _do_extract():
        return pdf_extractor.extract_pdf_content(pdf_path)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_do_extract)
            extracted = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise RuntimeError("PDF extraction timed out (over 2 minutes).")

    chunks = chunker.process_extracted_file(extracted)
    chunk_index = build_index.build_chunk_index(chunks)
    sections = section_rep_builder.build_section_reps(extracted["sections"], chunk_index)

    # Save to cache
    _save_cache(user_dir, pdf_basename, sections, chunk_index)
    return sections, chunk_index


def load_pdf(pdf_file, system_prompt, username):
    if not username:
        return None, None, "Please log in first."
    if pdf_file is None:
        return None, None, "Please upload a PDF."
    user_dir = ensure_user_dir(username)
    dest_path = os.path.join(user_dir, os.path.basename(pdf_file.name))
    shutil.copy(pdf_file.name, dest_path)
    try:
        sections, chunk_index = process_pdf(dest_path, user_dir)
        msg = f"Processed {os.path.basename(dest_path)}"
    except RuntimeError as e:
        return None, None, str(e)
    # Record upload & system prompt for this user and persist
    with _DB_LOCK:
        user_record = _USER_DB["users"].setdefault(
            username,
            {"password": USERS.get(username, hash_password("default")), "uploads": [], "prompts": []},
        )
        if dest_path not in user_record["uploads"]:
            user_record["uploads"].append(dest_path)
        if system_prompt and system_prompt not in user_record["prompts"]:
            user_record["prompts"].append(system_prompt)
        _save_user_db(_USER_DB)
    return sections, chunk_index, msg


# Helper to load an existing PDF by name for the user
def load_existing_pdf(selected_name, username):
    if not username:
        return None, None, "Please log in first."
    if not selected_name:
        return None, None, "No previous PDF selected."
    user_dir = ensure_user_dir(username)
    pdf_path = os.path.join(user_dir, selected_name)
    if not os.path.exists(pdf_path):
        return None, None, "File not found."

    # Attempt to load cached sections/index
    pdf_basename = os.path.splitext(selected_name)[0]
    sections, chunk_index = _load_cache(user_dir, pdf_basename)
    if sections is None or chunk_index is None:
        try:
            sections, chunk_index = process_pdf(pdf_path, user_dir)
            msg = f"Processed {selected_name}"
        except RuntimeError as e:
            return None, None, str(e)
    else:
        msg = f"Loaded cached data for {selected_name}"
    return sections, chunk_index, msg


def load_all_cached_pdfs(username):
    """
    Load sections + chunk index for every PDF the user has previously uploaded.
    Sections are concatenated in upload order; chunk indexes are merged.
    """
    if not username:
        return None, None, "Please log in first."
    user_dir = ensure_user_dir(username)
    uploads = _USER_DB["users"].get(username, {}).get("uploads", [])
    if not uploads:
        return None, None, "No cached PDFs were found."

    all_sections = []
    all_chunks = []
    for pdf_path in uploads:
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        sections, chunk_index = _load_cache(user_dir, pdf_basename)
        if sections is None or chunk_index is None:
            # Skip files that were never processed / cache missing
            continue
        # Tag each section with its source PDF name
        tagged_sections = []
        for sec in sections:
            sec_copy = sec.copy()
            sec_copy["file_name"] = pdf_basename  # add filename field
            tagged_sections.append(sec_copy)
        all_sections.extend(tagged_sections)
        all_chunks.extend(chunk_index)

    if not all_sections:
        return None, None, "No cached data found. Process PDFs first."

    msg = f"Loaded cached data for {len(all_sections)} sections across {len(uploads)} PDFs"
    return all_sections, all_chunks, msg


def delete_cached_pdf(selected_name, username):
    """
    Delete a previously uploaded PDF and its cached data for the user.
    Returns updated dropdown choices and a status message.
    """
    if not username:
        return None, None, gr.update(), "Please log in first."
    if not selected_name:
        return None, None, gr.update(), "No PDF selected."
    user_dir = ensure_user_dir(username)
    pdf_path = os.path.join(user_dir, selected_name)
    if not os.path.exists(pdf_path):
        return None, None, gr.update(), "File not found."

    # Remove pdf file
    try:
        os.remove(pdf_path)
    except OSError as e:
        return None, None, gr.update(), f"Delete failed: {e}"

    # Remove cached section/index files
    pdf_basename = os.path.splitext(selected_name)[0]
    sec_path, idx_path = _cache_paths(user_dir, pdf_basename)
    for p in (sec_path, idx_path):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass  # ignore

    # Update user DB
    with _DB_LOCK:
        uploads = _USER_DB["users"].get(username, {}).get("uploads", [])
        if pdf_path in uploads:
            uploads.remove(pdf_path)
        _save_user_db(_USER_DB)

    # Build new dropdown choices
    choices = [os.path.basename(u) for u in uploads]
    dropdown_update = gr.update(choices=choices, value=(choices[0] if choices else None))
    msg = f"Deleted {selected_name}" if choices is not None else "All PDFs deleted."
    return None, None, dropdown_update, msg


def ask_question(question, sections, chunk_index, system_prompt, username, use_index):
    fine_only = not use_index 
    if not username:
        return "Please log in first."
    if sections is None or chunk_index is None:
        return "Please upload and process a PDF first."
    prompt = system_prompt or DEFAULT_PROMPT
    bot = PDFChatBot(sections, chunk_index, system_prompt=prompt)
    answer = bot.answer(question, fine_only=fine_only)
    answer = answer.replace('<|endoftext|><|im_start|>user',"=== System Prompt ===")
    answer = answer.replace('<|im_end|>\n<|im_start|>assistant','')
    answer = answer.replace('<|im_end|>','')
    answer_output = answer.split("=== Answer ===")[-1].strip()
    reference_output = answer.split("=== User Question ===")[0].strip().split("=== Document Context ===")[-1].strip()

    return answer_output, reference_output

def handle_pdf(pdf_file, subject):
    if not pdf_file:
        return "PDF 파일을 업로드해주세요."
    result = pdf_extractor.extract_pdf_content(pdf_file.name)
    summary = concept_summary.summarize_concepts_from_pdf(result,subject)
    return summary

def generate(pdf_file, subject):
    if not pdf_file:
        return "PDF 파일을 업로드해주세요."
    result = pdf_extractor.extract_pdf_content(pdf_file.name)
    summary = summarize_concepts_from_questions(result)
    questions = generate_questions_from_concepts(summary.splitlines(), subject)
    return questions

def grade(image, qa_text, subject):
    if not image:
        return "", "답안 이미지를 업로드해주세요."
    student_text = extract_text_from_image(image.name)
    qa_list = parse_generated_questions(qa_text)
    # 예시: 첫 번째 문제만 채점 대상으로 활용

    if qa_list:
        reference_question = qa_list[0]["question"]
        reference_answer = qa_list[0]["answer"]
    else:
        return "", "생성된 문제가 없습니다."
    
    # subject와 함께 LLM 채점
    feedback = grade_answer(student_text, reference_answer, subject,question=reference_question)
    return student_text, feedback

with gr.Blocks(theme=gr.themes.Soft(), css="""
body {
    font-family: 'Helvetica Neue', sans-serif;
    background-color: #f0fdf4;
    color: #1f2937;
}
h1, h2, h3 {
    color: #065f46;
}
textarea, input, select {
    border: 1px solid #d1fae5;
    padding: 0.5rem;
    border-radius: 0.375rem;
}
button {
    background-color: #10b981;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    font-weight: bold;
    border: none;
    cursor: pointer;
}
button:hover {
    background-color: #059669;
}
input[type="text"], input[type="password"] {
    background-color: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
}
""") as demo:
     
    gr.Markdown("""
    #📘 Edeuron 
    ##- Understand more, Predict better
    중고등학생을 위한 AI 기반 시험 문제 요약, 문제 생성, 답안 채점 서비스를 제공합니다.""")

    # Login components
    with gr.Column():
        with gr.Row():
            login_user = gr.Textbox(label="Username")
            login_pass = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login", variant="primary")
        login_status = gr.Textbox(label="Login Status", interactive=False)

    # Main interaction area – hidden until login is successful
    with gr.Column(visible=False) as main_area:
        # First row ‑‑ two side‑by‑side upload panels
        with gr.Row():
            # Left column – previously uploaded files
            with gr.Column(scale=1):  
                gr.Markdown("""
                    ---
                    ### 📂 지금까지 올린 파일 보기
                    - 아래 목록에서 파일을 선택하면 다시 열 수 있어요.
                    - **선택해서 열기**: 목록에서 하나를 골라서 다시 불러와요.
                    - **모두 불러오기**: 지금까지 올린 모든 파일을 다 가져와요.
                    - **삭제하기**: 선택한 파일과 내용을 완전히 지워요.
                    """)

                existing_dropdown = gr.Dropdown(label="PDF 확인하기", choices=[])
                load_existing_btn = gr.Button("선택해서 열기", variant="primary")
                load_all_btn = gr.Button("모두 불러오기", variant="secondary")
                delete_btn = gr.Button("삭제하기", variant="stop")

            # Right column – new upload
            with gr.Column(scale=1): 
                gr.Markdown("###🆕 새로운 파일 올리기")
                gr.Markdown("- 새로운 파일을 올리세요. 시간은 **약 2초** 정도 소요됩니다.")
                gr.Markdown("- **파일 올리기** 를 누르면 PDF파일이 올라갑니다.")
                gr.Markdown("- 파일은 동일한 이름의 유저/비밀번호를 가진 사람일 경우, 유지됩니다.")

                pdf_input = gr.File(label="PDF File", file_types=[".pdf"])
                load_btn = gr.Button("파일 올리기", variant="primary")

            with gr.Column(scale=1): 
                gr.Markdown("###✔️ 과목 선택하기")
                gr.Markdown("-올린 파일의 과목을 선택해주세요.")
                gr.Markdown("- **과목 선택** 을 누르면, 파일의 과목이 결정됩니다")
                gr.Markdown("- 과목을 기준으로 예상 문제를 생성하니, 올바른 과목을 선택해주세요!.")

                subject = gr.Dropdown(label="과목 선택", choices=["국어", "영어", "수학", "사회", "과학"])

        # Second row ‑‑ status spans the full width
        status = gr.Textbox(label="PDF Status", interactive=False)

        DEFAULT_PROMPTS = {
            "국어": "너는 친절한 국어 선생님이야. 문서 내용을 참고해서 질문에 대해 핵심 개념과 예시를 들어 설명해줘. 모르면 모른다고 해줘.",
            "영어": "너는 친절한 영어 선생님이야. 문서 내용을 바탕으로 학생이 이해하기 쉽게 문법이나 어휘를 설명해줘.",
            "수학": "너는 수학을 잘 가르치는 선생님이야. 문서 내용을 참고해서 질문에 대해 단계별로 풀이해줘. 개념과 간단한 예제를 함께 설명해줘.",
            "사회": "너는 사회를 쉽게 설명해주는 선생님이야. 문서 내용을 바탕으로 개념과 배경을 학생 눈높이에 맞춰 설명해줘.",
            "과학": "너는 과학을 재밌고 쉽게 알려주는 선생님이야. 질문에 대해 문서 내용을 토대로 핵심 개념과 원리를 쉽게 설명해줘."
        }

        gr.Markdown("### Edeuron AI teacher's mode")
        prompt_input = gr.Textbox(label="mode", lines=5, value="")
        def update_prompt(s):
            return DEFAULT_PROMPTS.get(s, "")
        subject.change(fn=update_prompt, inputs=subject, outputs=prompt_input)

        gr.Markdown("### 📚 개념 요약")
        concept_summary = gr.Textbox(label="요약", lines=10)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Edeuron AI Teacher")
                gr.Markdown("- 👩‍🏫 올린 파일에 기반하여, **문제 번호**와 모르는 개념에 대해 질문하세요.")
                gr.Markdown("- Check **Coarse-to-Fine Search** to enable Table of Contents based search.")
                question_input = gr.Textbox(label="❔Question")
                use_index = gr.Checkbox(label="Coarse-to-Fine Search", value=False)
                ask_btn = gr.Button("📣알려주세요!", variant="primary")

        gr.Markdown("### 답변")
        answer_output = gr.Textbox(label="👩‍🏫 선생님은 이렇게 생각해요", interactive=False, lines=10)

        gr.Markdown("### 🖍 예상 문제")
        generate_btn = gr.Button("예상 문제 생성")
        generated_questions = gr.Textbox(label="🖍 예상 문제", lines=10)

        with gr.Row():
            # Left column 
            with gr.Column(scale=1):  
                gr.Markdown("### ✅ 정답 입력")
                answer_image = gr.File(label="✍ 손글씨 답안 이미지 업로드", file_types=["image/*"])
                correct_answer = gr.Textbox(label="정답 입력", placeholder="모범 답안을 입력하세요", lines=4)

            # Right column 
            with gr.Column(scale=1): 
                gr.Markdown("### 📄 답안 추출")
                extracted_answer = gr.Textbox(label="✏️ 추출된 답안", lines=2)

 
        grade_btn = gr.Button("💯 채점하기")
        evaluation = gr.Textbox(label="📊 피드백", lines=3)

        gr.Markdown("### 참조 개념")
        reference_output = gr.Textbox(label="References", interactive=False, lines=10)

    

    logged_in_state = gr.State(False)
    username_state = gr.State("")
    sections_state = gr.State()
    index_state = gr.State()
    pdf_input.change(fn=handle_pdf, inputs=[pdf_input, subject], outputs=concept_summary)
    generate_btn.click(fn=generate, inputs=[pdf_input, subject], outputs=generated_questions)
    grade_btn.click(fn=grade, inputs=[answer_image, generated_questions, subject], outputs=[extracted_answer, evaluation])
    
    login_btn.click(
        login_and_prepare,
        inputs=[login_user, login_pass],
        outputs=[logged_in_state, username_state, login_status, main_area, prompt_input, existing_dropdown],
    )
    load_btn.click(load_pdf, inputs=[pdf_input, prompt_input, username_state], outputs=[sections_state, index_state, status])
    load_existing_btn.click(
        load_existing_pdf,
        inputs=[existing_dropdown, username_state],
        outputs=[sections_state, index_state, status]
    )
    load_all_btn.click(
        load_all_cached_pdfs,
        inputs=[username_state],
        outputs=[sections_state, index_state, status]
    )
    delete_btn.click(
        delete_cached_pdf,
        inputs=[existing_dropdown, username_state],
        outputs=[sections_state, index_state, existing_dropdown, status]
    )
    question_input.submit(ask_question, inputs=[question_input, sections_state, index_state, prompt_input, username_state, use_index], outputs=[answer_output, reference_output])
    ask_btn.click(ask_question, inputs=[question_input, sections_state, index_state, prompt_input, username_state, use_index], outputs=[answer_output, reference_output])

if __name__ == "__main__":            
    demo.launch(server_name="0.0.0.0", server_port=30987)