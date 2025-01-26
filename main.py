import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import sys
import logging
from modules.YouTubeDownloader import YouTubeDownloader
from modules.AudioTranscriber import AudioTranscriber
from modules.TranscriptionAnalyzer import TranscriptionAnalyzer
from modules.VideoProcessor import VideoProcessor

@dataclass
class ProcessingConfig:
    """Configuration settings for video processing"""
    download: bool = False
    language: str = "en"
    num_highlights: int = 10
    min_length: int = 15
    max_length: int = 40
    add_subtitles: bool = False
    subtitles_language: str = "en"
    subtitle_position: str = "bottom"
    use_additional_video: bool = True

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_yes_no_input(prompt: str, default: bool = False) -> bool:
    """Get yes/no input from user with default value"""
    default_str = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not response:
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'")

def get_int_input(prompt: str, default: int, min_value: int = None, max_value: int = None) -> int:
    """Get integer input from user with validation"""
    while True:
        response = input(f"{prompt} [{default}]: ").strip()
        if not response:
            return default
        try:
            value = int(response)
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}")
                continue
            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")

def get_language_input(prompt: str, default: str) -> str:
    """Get language code input from user"""
    while True:
        response = input(f"{prompt} [{default}]: ").strip().lower()
        if not response:
            return default
        if len(response) == 2:  # Simple validation for ISO language codes
            return response
        print("Please enter a valid 2-letter language code (e.g., 'en', 'ru')")

def get_position_input(prompt: str, default: str) -> str:
    """Get subtitle position input from user"""
    valid_positions = ['bottom', 'center', 'top']
    while True:
        response = input(f"{prompt} ({'/'.join(valid_positions)}) [{default}]: ").strip().lower()
        if not response:
            return default
        if response in valid_positions:
            return response
        print(f"Please enter one of: {', '.join(valid_positions)}")

def get_interactive_config() -> ProcessingConfig:
    """Get configuration through interactive user input"""
    print("\nPlease provide processing configuration:")
    print("(Press Enter to use default values)\n")
    
    config = ProcessingConfig()
    
    config.download = get_yes_no_input("Download video from YouTube?", False)
    config.language = get_language_input("Transcription language (ISO code)", "en")
    config.num_highlights = get_int_input("Number of highlights", 10, 1, 100)
    config.min_length = get_int_input("Minimum highlight length (seconds)", 15, 1)
    config.max_length = get_int_input("Maximum highlight length (seconds)", 40)
    
    if config.min_length >= config.max_length:
        print("Maximum length must be greater than minimum length. Using defaults.")
        config.min_length = 15
        config.max_length = 40
    
    config.add_subtitles = get_yes_no_input("Add subtitles to output videos?", False)
    
    if config.add_subtitles:
        config.subtitles_language = get_language_input("Subtitles language (ISO code)", "en")
        config.subtitle_position = get_position_input("Subtitle position", "bottom")
    
    config.use_additional_video = get_yes_no_input("Use additional video overlay?", True)
    
    return config

def parse_arguments() -> ProcessingConfig:
    """Parse command line arguments and return configuration"""
    parser = argparse.ArgumentParser(description='Video processing and highlight extraction tool')
    
    parser.add_argument('--download', action='store_true', help='Download video from YouTube')
    parser.add_argument('--language', help='Language for transcription (ISO code)')
    parser.add_argument('--num_highlights', type=int, help='Number of highlights to extract')
    parser.add_argument('--min_length', type=int, help='Minimum highlight length in seconds')
    parser.add_argument('--max_length', type=int, help='Maximum highlight length in seconds')
    parser.add_argument('--add_subtitles', action='store_true', help='Add subtitles to output videos')
    parser.add_argument('--subtitles_language', help='Language for subtitles (ISO code)')
    parser.add_argument('--subtitle_position', 
                       choices=['bottom', 'center', 'top'],
                       help='Position of subtitles')
    parser.add_argument('--use_additional_video', 
                       action='store_true',
                       help='Use additional video overlay')

    args = parser.parse_args()
    
    # If no command line arguments provided, use interactive input
    if not any(vars(args).values()):
        return get_interactive_config()
    
    # Otherwise, create config from provided arguments, using defaults for missing ones
    config = ProcessingConfig()
    for key, value in vars(args).items():
        if value is not None:  # Only override if argument was provided
            setattr(config, key, value)
    
    return config

def get_video_path(config: ProcessingConfig) -> str:
    """Get video path either by downloading or using existing file"""
    videos_dir = Path.cwd() / "videos"
    videos_dir.mkdir(exist_ok=True)

    if config.download:
        try:
            downloader = YouTubeDownloader(str(videos_dir))
            link = input("YouTube link: ")
            return downloader.download_video(link, "1080p")
        except Exception as e:
            logging.error(f"Failed to download video: {e}")
            sys.exit(1)
    else:
        video_name = input("Video name with extension: ")
        return str(videos_dir / video_name)

def transcribe_video(video_path: str, language: str) -> str:
    """Transcribe video and return path to transcription file"""
    logging.info("Starting transcription...")
    
    transcriber = AudioTranscriber(
        model_size="base",
        device="cpu",
        compute_type="int8"
    )
    
    if not transcriber._check_audio_stream(video_path):
        logging.error("No audio stream found in the video file")
        sys.exit(1)
        
    segments = transcriber.transcribe_video(
        video_path=video_path,
        language=language,
        beam_size=4
    )
    
    # Log segments
    for segment in segments:
        logging.info(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")
        
    return transcriber.save_transcription(segments, "transcription.json")

def analyze_transcription(transcription_path: str, config: ProcessingConfig) -> list:
    """Analyze transcription and return highlights"""
    logging.info("Starting analysis...")
    
    analyzer = TranscriptionAnalyzer()
    
    try:
        with open(transcription_path, 'r', encoding='utf-8') as file:
            original_transcription = json.load(file)
            
        transcription_json = json.dumps(original_transcription)
        
        highlights = analyzer.get_highlights(
            transcription=transcription_json,
            num_highlights=config.num_highlights,
            min_length=config.min_length,
            max_length=config.max_length
        )
        
        # Log results
        for idx, highlight in enumerate(highlights, start=1):
            logging.info(f"\nHighlight {idx}:")
            logging.info(f"Timestamp: {highlight.start:.2f}s to {highlight.end:.2f}s")
            logging.info(f"Duration: {highlight.end - highlight.start:.2f}s")
            logging.info(f"Content: {highlight.content}")
            
        return highlights
        
    except Exception as e:
        logging.error(f"Failed to analyze transcription: {e}")
        sys.exit(1)

def process_highlights(video_path: str, highlights: list, config: ProcessingConfig):
    """Process video highlights"""
    logging.info("Starting video processing...")
    
    for idx, highlight in enumerate(highlights, start=1):
        try:
            processor = VideoProcessor()
            logging.info(f"Processing highlight {idx}/{len(highlights)}")
            logging.info(f"Time range: {highlight.start:.2f}s to {highlight.end:.2f}s")
            
            output_path = processor.process_video(
                input_path=video_path,
                start_time=highlight.start,
                end_time=highlight.end,
                language=config.subtitles_language,
                add_subtitles=config.add_subtitles,
                use_additional_video=config.use_additional_video,
                subtitle_position=config.subtitle_position
            )
            logging.info(f"Processed highlight saved to: {output_path}")
            
        except Exception as e:
            logging.error(f"Failed to process highlight {idx}: {e}")
            continue

def main():
    """Main execution function"""
    setup_logging()
    config = parse_arguments()
    
    try:
        video_path = get_video_path(config)
        transcription_path = transcribe_video(video_path, config.language)
        highlights = analyze_transcription(transcription_path, config)
        process_highlights(video_path, highlights, config)
        logging.info("Processing completed successfully!")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()