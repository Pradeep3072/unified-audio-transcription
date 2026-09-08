from pydantic import BaseModel, Field
from typing import List, Optional


class EvaluationMatrix(BaseModel):
    grammar_score: int = Field(description="Grammar and clarity score of the original text from 1 to 10")
    readability_improvement: str = Field(description="Brief explanation of how readability and context were improved")
    changes_made: List[str] = Field(description="List of specific corrections made to the original text")


class CorrectedTranscript(BaseModel):
    corrected_text: str = Field(description="The grammatically corrected and punctuated transcript")
    summary: str = Field(description="A concise summary of the transcribed text capturing the main points")
    evaluation: EvaluationMatrix


class MetricsResult(BaseModel):
    audio_duration_sec: float
    transcription_time_ms: float
    real_time_factor: float
    word_count: int
    words_per_minute: float
    wer: Optional[float] = None
    cer: Optional[float] = None
