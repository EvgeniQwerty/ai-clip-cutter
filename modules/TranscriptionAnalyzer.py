import os
import json
import re
from typing import Union, List, Dict
from dataclasses import dataclass
from pathlib import Path
import requests
from dotenv import load_dotenv

@dataclass
class HighlightSegment:
    """Class representing a highlighted video segment."""
    start: float
    end: float
    content: str

@dataclass
class MistralConfig:
    """Configuration settings for Mistral API."""
    api_key: str
    api_url: str = "https://api.mistral.ai/v1/chat/completions"
    model: str = "open-mistral-nemo"
    temperature: float = 0.7

class TranscriptionAnalyzer:    
    """Class for analyzing video transcriptions to identify highlight segments."""

    #доделать. Почему-то выбирает куски один за другим. И время берёт не из json

    SYSTEM_PROMPT_TEMPLATE = """
    Analyze the provided video transcript JSON and identify the {num_highlights} most interesting moments based on their content. 
    These highlights should have a duration between {min_length} and {max_length} seconds.
    
    ### Input Data Example (DO NOT USE THE TIMESTAMPS OR TEXT BELOW IN YOUR OUTPUT):
        [
            {{
                "text": "It doesn't matter how high school you take and how high school you are, life is always a way to overcome your year.",
                "start": 56.62,
                "end": 65.12
            }},
            {{
                "text": "But it doesn't matter how much you get from life, where are you most important conclusions from these gifts in the future?",
                "start": 65.12,
                "end": 74.68
            }},
            {{
                "text": "What conclusions have I done?",
                "start": 76.68,
                "end": 77.60
            }},
            {{
                "text": "Let's find out!",
                "start": 77.70,
                "end": 100.62
            }}
        ]

    ### Explanation of Input Data:
        - Each object in the input JSON contains a "text" fragment along with its exact "start" and "end" timestamps.
        - **The example data above is provided only to demonstrate the format of the input. Under no circumstances should the timestamps or text from this example be used in your output.**

    ### Expected Output Data (DO NOT USE THESE VALUES; ONLY FOLLOW THE FORMAT):
        [
            {{
                "start": 56.62,
                "content": "It doesn't matter how high school you take and how high school you are, life is always a way to overcome your year. But it doesn't matter how much you get from life, where are you most important conclusions from these gifts in the future?",
                "end": 74.68
            }},
            {{
                "start": 77.70,
                "content": "What conclusions have I done?",
                "end": 100.62
            }}
        ]

    ### Explanation of Output Data:
        1. The "start" field in the output object is **exactly** the "start" value of the first input JSON object included in the highlight.
        2. The "end" field in the output object is **exactly** the "end" value of the last input JSON object included in the highlight.
        3. The "content" field is the concatenated "text" values from all input objects included in the highlight.

    ### Rules:
        1. **Do not invent or modify timestamps.** Only use the "start" and "end" values from the input JSON.
        2. The range of "start" and "end" in the output must match a contiguous sequence of input JSON objects.
        3. Ensure that the total duration of each highlight (end - start) falls within {min_length} and {max_length}.
        4. Highlights **must be selected from different parts of the input JSON.** If two or more highlights are chosen from the same section or are sequential in the JSON, the output will be considered invalid.
        5. Provide exactly {num_highlights} output objects.
        6. Do not add any explanations or commentary in your response. Only return the JSON output.

    ### Important Note:
        - If you generate any timestamps that are not present in the input JSON, or if the highlights are clustered in the same part of the JSON, the output will be invalid, and a penalty will be applied.
        - Distribute the highlights evenly across the input JSON to ensure they represent diverse and meaningful segments of the content.
    """

    def __init__(self):
        """
        Initialize the analyzer with API configuration.
        Raises:
            ValueError: If the API key is not set.
        """
        load_dotenv()
        self.config = MistralConfig(
            api_key=os.getenv("MISTRAL_API")
        )
        if not self.config.api_key:
            raise ValueError("MISTRAL_API environment variable is not set")

    def load_transcription(self, file_path: Union[str, Path]) -> str:
        """
        Load transcription data from a JSON file.

        Args:
            file_path: Path to the JSON file containing transcription data.

        Returns:
            str: Combined transcription text.

        Raises:
            ValueError: If the file is not found or contains invalid JSON.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"Transcription file not found: {file_path}")

        try:
            with file_path.open('r', encoding='utf-8') as file:
                data = json.load(file)
                return ' '.join(segment["text"] for segment in data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in transcription file: {e}")

    """
    Coz Mistral can't correctly handle timestamps we do it manually 
    """
    def _fix_highlight_timestamps(self, highlight: Dict, original_transcription: List[Dict]) -> Dict:
        """
        Fix timestamps for a single highlight by matching its first and last sentences
        with the original transcription.

        Args:
            highlight: Dictionary containing highlight information
            original_transcription: List of original transcription segments

        Returns:
            Dict: Updated highlight with correct timestamps
        """
        # Split content into sentences
        sentences = re.split(r'(?<=[.!?])\s+', highlight['content'].strip())
        sentences = [s for s in sentences if s]  # Remove empty sentences
        
        if not sentences:
            return highlight

        # Find first sentence in original transcription
        first_sentence = sentences[0]
        last_sentence = sentences[-1]
        
        # Find matching segments
        start_segment = None
        end_segment = None
        
        for segment in original_transcription:
            if first_sentence in segment['text'] and start_segment is None:
                start_segment = segment
            if last_sentence in segment['text']:
                end_segment = segment
            
            if start_segment and end_segment:
                break
        
        # Update timestamps if matches were found
        if start_segment:
            highlight['start'] = float(start_segment['start'])
        if end_segment:
            highlight['end'] = float(end_segment['end'])
            
        return highlight

    def _extract_highlights(self, response: str, original_transcription: List[Dict]) -> List[HighlightSegment]:
        """
        Extract multiple highlights from the API response and fix their timestamps.

        Args:
            response: JSON string from the API response
            original_transcription: List of original transcription segments

        Returns:
            List[HighlightSegment]: List of highlighted segments with corrected timestamps

        Raises:
            ValueError: If the response is invalid or improperly formatted
        """
        try:
            data = json.loads(response)
            
            # Fix timestamps for each highlight
            fixed_data = [
                self._fix_highlight_timestamps(highlight, original_transcription)
                for highlight in data
            ]
            
            # Convert to HighlightSegment objects
            highlights = [
                HighlightSegment(
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    content=segment["content"]
                ) for segment in fixed_data
            ]
            return highlights
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"Error parsing API response: {e}")

    def get_highlights(self, transcription: str, num_highlights: int = 1, min_length: int = 5, max_length: int = 60) -> List[HighlightSegment]:
        """
        Get multiple highlight segments using the Mistral API.

        Args:
            transcription: Full transcription text
            num_highlights: Number of highlight segments to retrieve
            min_length: Minimum length of a highlight in seconds
            max_length: Maximum length of a highlight in seconds

        Returns:
            List[HighlightSegment]: List of highlighted segments

        Raises:
            ValueError: If the API request fails or response is invalid
        """
        # Parse the original transcription to have access to the segments
        original_transcription = json.loads(transcription)
        
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            num_highlights=num_highlights,
            min_length=min_length,
            max_length=max_length
        )

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcription}
            ],
            "temperature": self.config.temperature
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            json_string = result["choices"][0]["message"]["content"]
            return self._extract_highlights(json_string, original_transcription)

        except requests.RequestException as e:
            raise ValueError(f"API request failed: {e}")
        except ValueError as e:
            raise ValueError(f"Error processing API response: {e}")

if __name__ == "__main__":
    try:
        # Initialize analyzer
        analyzer = TranscriptionAnalyzer()
        transcription_path = "transcription.json"
        
        # Load and parse the original transcription
        with open(transcription_path, 'r', encoding='utf-8') as file:
            original_transcription = json.load(file)
            
        # Convert the transcription to the format expected by the API
        transcription_json = json.dumps(original_transcription)
        
        # Get highlights with corrected timestamps
        highlights = analyzer.get_highlights(
            transcription=transcription_json,
            num_highlights=6,
            min_length=30,
            max_length=70
        )
        
        # Print results
        print("\nAnalysis Results:")
        print("-" * 50)
        for idx, highlight in enumerate(highlights, start=1):
            print(f"Highlight {idx}:")
            print(f"Timestamp: {highlight.start:.2f}s to {highlight.end:.2f}s")
            print(f"Duration: {highlight.end - highlight.start:.2f}s")
            print(f"Content: {highlight.content}")
            print("-" * 50)
            
    except FileNotFoundError:
        print(f"Error: Could not find transcription file at {transcription_path}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in transcription file")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
