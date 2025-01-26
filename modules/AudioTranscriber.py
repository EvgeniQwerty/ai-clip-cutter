import os
from typing import List, Union
from pathlib import Path
import json
import subprocess
from dataclasses import dataclass
from faster_whisper import WhisperModel
import ffmpeg

# Set environment variables to handle OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Disable parallelism in ffmpeg to avoid conflicts
os.environ["FFTW_WISDOM_ONLY"] = "TRUE"

@dataclass
class TranscriptionSegment:
    """Class representing a single transcription segment."""
    text: str
    start: float
    end: float

class AudioTranscriber:
    """Class for transcribing audio from video files."""
    
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize the transcriber.
        
        Args:
            model_size: Size of the Whisper model to use 
                       ("tiny", "base", "small", "medium", "large-v2")
            device: Device to use for inference ("cpu" or "cuda")
            compute_type: Model computation type ("int8", "float16", or "float32")
        """
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=4  # Limit CPU threads to avoid conflicts
        )
    
    def _check_audio_stream(self, video_path: Union[str, Path]) -> bool:
        """
        Check if video file has an audio stream.
        
        Args:
            video_path: Path to video file
            
        Returns:
            bool: True if video has audio stream, False otherwise
        """
        try:
            probe = ffmpeg.probe(str(video_path))
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
            return len(audio_streams) > 0
        except ffmpeg.Error:
            return False

    def _extract_audio(self, video_path: Union[str, Path], output_path: Union[str, Path]) -> Path:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            output_path: Path to save extracted audio
            
        Returns:
            Path: Path to extracted audio file
            
        Raises:
            ValueError: If video has no audio stream or extraction fails
        """
        if not self._check_audio_stream(video_path):
            raise ValueError("No audio stream found in the video file")

        try:
            command = [
                'ffmpeg',
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-threads', '1',  # Use single thread
                '-y',  # Overwrite output file
                str(output_path)
            ]
            
            process = subprocess.run(
                command,
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                raise ValueError(f"FFmpeg error: {process.stderr}")
                
            return Path(output_path)
            
        except subprocess.SubprocessError as e:
            raise ValueError(f"Error running FFmpeg: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error extracting audio: {str(e)}")

    def transcribe_video(self, 
                        video_path: Union[str, Path],
                        language: str,
                        temp_dir: Union[str, Path] = None,
                        beam_size: int = 1) -> List[TranscriptionSegment]:
        """
        Transcribe audio from video file.
        
        Args:
            video_path: Path to video file
            language: Language code in ISO format (e.g., 'en', 'ru', 'es')
            temp_dir: Directory for temporary files. Default is video file directory
            beam_size: Beam size for transcription (smaller = faster but potentially less accurate)
            
        Returns:
            List[TranscriptionSegment]: List of transcription segments with text and timestamps
            
        Raises:
            ValueError: If error occurs during transcription or file processing
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")
            
        # Set up temporary directory
        temp_dir = Path(temp_dir) if temp_dir else video_path.parent
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True)
            
        # Extract audio to WAV
        audio_path = temp_dir / f"{video_path.stem}_audio.wav"
        try:
            self._extract_audio(video_path, audio_path)
            
            # Transcribe audio with reduced complexity
            segments, _ = self.model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,  # Reduced beam size
                vad_filter=True,
                word_timestamps=False  # Disable word timestamps for better performance
            )
            
            # Convert segments to our format
            transcription = []
            for segment in segments:
                transcription.append(TranscriptionSegment(
                    text=segment.text.strip(),
                    start=segment.start,
                    end=segment.end
                ))
            
            return transcription
            
        except Exception as e:
            raise ValueError(f"Error during transcription: {str(e)}")
            
        finally:
            # Clean up temporary audio file
            if audio_path.exists():
                audio_path.unlink()

    def save_transcription(self, segments: List[TranscriptionSegment], output_path: Union[str, Path]):
        """
        Save transcription segments to a JSON file.
        
        Args:
            segments: List of transcription segments
            output_path: Path to save the JSON file
        """
        output_path = Path(output_path)
        data = [
            {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            }
            for segment in segments
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return output_path    

if __name__ == "__main__":
    # Example usage with optimized settings for Spyder/Anaconda environment
    try:
        transcriber = AudioTranscriber(
            model_size="base",
            device="cpu",
            compute_type="int8"  # Use int8 quantization for better compatibility
        )
        
        video_path = f"{Path.cwd()}/videos/video.mp4"
        
        if not transcriber._check_audio_stream(video_path):
            print("Error: No audio stream found in the video file")
            exit(1)
            
        segments = transcriber.transcribe_video(
            video_path=video_path,
            language="en",
            beam_size=2  # Use minimal beam size for testing
        )
        
        # Print segments
        for segment in segments:
            print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")
            
        # Save to JSON
        transcriber.save_transcription(segments, "transcription.json")
            
    except ValueError as e:
        print(f"Error: {e}")