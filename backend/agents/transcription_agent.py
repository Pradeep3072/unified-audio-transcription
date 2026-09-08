import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from backend.models.schemas import CorrectedTranscript

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
load_dotenv(dotenv_path)


class TranscriptionCorrectionAgent:
    """
    AI agent that corrects grammar/punctuation, summarizes, and evaluates
    a raw transcription using LangChain + Groq LLM.
    """

    def __init__(self):
        print("[Agent] Initializing Transcription Correction Agent...")
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            print("[Agent] WARNING: GROQ_API_KEY is not set. Agent correction will be skipped.")
            self.structured_llm = None
        else:
            self.llm = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0.2,
                api_key=self.api_key
            )
            self.structured_llm = self.llm.with_structured_output(CorrectedTranscript)
            print("[Agent] Ready.")

    def correct(self, raw_text: str, language: str = None) -> dict:
        if not raw_text or not raw_text.strip():
            return {"corrected_text": "", "summary": "", "evaluation": None}

        if not self.structured_llm:
            print("[Agent] Skipping correction — GROQ_API_KEY is missing.")
            return {"corrected_text": raw_text, "summary": "", "evaluation": None}

        lang_hint = f" The expected language is '{language}'." if language else ""
        if language == "tanglish":
            lang_hint += (
                " 'Tanglish' means a mix of Tamil and English. The raw text may contain a messy mix "
                "of Tamil script and English script. Read both carefully, and rewrite the entire "
                "sentence into clean, natural-sounding phonetic Tanglish using ONLY the English (Latin) alphabet."
            )

        prompt = f"""
You are an expert polyglot audio transcription proofreader.
Review the following raw transcription.{lang_hint} First, identify the language of the text.
Then, fix any spelling, grammatical, or punctuation errors in that specific language.
Ensure that the corrections follow the grammar rules of the detected language.
Keep the corrected text in the ORIGINAL language. Do NOT translate it to English.
Do NOT add external information or change the original meaning of the speaker.
Then, provide a concise summary of the transcribed text capturing its main points.
Finally, evaluate your own corrections.

Raw Transcription:
{raw_text}
"""

        try:
            print("[Agent] Sending transcript to Groq for correction...")
            result = self.structured_llm.invoke(prompt)
            if hasattr(result, "model_dump"):
                return result.model_dump()
            elif hasattr(result, "dict"):
                return result.dict()
            elif isinstance(result, dict):
                return result
            else:
                return {"corrected_text": raw_text, "summary": "", "evaluation": None}
        except Exception as e:
            print(f"[Agent] Error: {e}")
            return {
                "corrected_text": raw_text,
                "summary": f"⚠️ AI Correction Failed: {e}",
                "evaluation": None
            }
