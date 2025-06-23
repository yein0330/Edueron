import pytesseract
from PIL import Image
from difflib import SequenceMatcher

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from a handwritten or printed answer image using OCR.
    """
    image = Image.open(image_path).convert("L")  # convert to grayscale
    text = pytesseract.image_to_string(image, lang="eng+kor")  # or just "eng"
    return text.strip()