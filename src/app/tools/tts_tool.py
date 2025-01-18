from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import openai
from datetime import datetime
import os


class TextToSpeechToolInput(BaseModel):
    """Input schema for TextToSpeechTool."""
    text: str = Field(..., description="The text to convert to speech")
    voice: str = Field(default="alloy", description="The voice to use (alloy, echo, fable, onyx, nova, or shimmer)")

class TextToSpeechTool(BaseTool):
    name: str = "Text to Speech"
    description: str = (
        "Converts text to speech using OpenAI's TTS API and saves it as an audio file"
    )
    args_schema: Type[BaseModel] = TextToSpeechToolInput

    def _run(self, text: str, voice: str = "alloy") -> str:
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
        # Generate audio
        response = openai.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        # Save to file
        timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%p")
        output_file = f"output/speech_{timestamp}.mp3"
        response.stream_to_file(output_file)
        
        return f"Audio saved to {output_file}"
