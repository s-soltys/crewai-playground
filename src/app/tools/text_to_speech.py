from typing import Optional, ClassVar
import os
from openai import OpenAI
from crewai.tools import BaseTool
from datetime import datetime

class TextToSpeechTool(BaseTool):
    name: ClassVar[str] = "Text to Speech"
    description: ClassVar[str] = "Converts text to speech using OpenAI's Text-to-Speech API"

    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _execute(self, text: str, voice: str = "alloy") -> str:
        """
        Convert text to speech and save as an audio file.
        
        Args:
            text: The text to convert to speech
            voice: The voice to use (alloy, echo, fable, onyx, nova, or shimmer)
            
        Returns:
            str: Path to the generated audio file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%p")
        output_file = f"output/speech_{timestamp}.mp3"
        
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        response.stream_to_file(output_file)
        return f"Audio file generated at: {output_file}" 