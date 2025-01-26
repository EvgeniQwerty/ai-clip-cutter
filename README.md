# AI Clip Cutter

AI Clip Cutter is a Python-based tool designed to process videos, extract highlights, and enhance them with features like subtitles and additional overlays. This tool leverages transcription and analysis to automatically find and create engaging video segments.

## Features

- **YouTube Video Download:** Download videos directly from YouTube using a link.
- **Automatic Transcription:** Generate text transcriptions from video audio using AI.
- **Highlight Extraction:** Automatically identify the most engaging highlights based on transcription analysis.
- **Customizable Settings:** Adjust the number, duration, and language of highlights.
- **Subtitle Integration:** Add subtitles to highlights in multiple languages and customize their position.
- **Overlay Support:** Optionally apply additional video overlays for enhanced visuals.

## Why AI Clip Cutter?

I created this tool after trying several existing GitHub solutions, none of which worked for my needs. This project aims to fill that gap by providing a robust and user-friendly solution for video highlight extraction.

## Limitations and Testing Environment

- **Testing Environment:** The script has only been tested on an Anaconda environment with Python 3.11 in the Spyder IDE. It may encounter issues in other setups.
- **Optimization:** Currently, transcription is performed twice to support languages other than English. For English videos, this can be optimized by simplifying the logic, as most content is in English.
- **Highlight Count Variability:** The AI (Mistral) used for transcription analysis sometimes generates a different number of highlights than requested. While this is unavoidable with the current model, switching to OpenAI's API could improve consistency—though it involves additional costs.

## Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd AI-Clip-Cutter
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command-Line Arguments
Run the script with command-line arguments for non-interactive usage:

```bash
python main.py [options]
```

| Option                  | Description                                |
|-------------------------|--------------------------------------------|
| `--download`            | Download video from YouTube.              |
| `--language`            | Language for transcription (ISO code).    |
| `--num_highlights`      | Number of highlights to extract.          |
| `--min_length`          | Minimum highlight length in seconds.      |
| `--max_length`          | Maximum highlight length in seconds.      |
| `--add_subtitles`       | Add subtitles to output videos.           |
| `--subtitles_language`  | Language for subtitles (ISO code).        |
| `--subtitle_position`   | Subtitle position (`bottom`, `center`, `top`). |
| `--use_additional_video`| Use additional video overlay.             |

### Interactive Mode
If no arguments are provided, the script will guide you through an interactive configuration process.

```bash
python main.py
```

## Example Workflow

1. Provide a YouTube video link or a local video file.
2. Choose transcription and highlight settings (e.g., language, number of highlights).
3. Optionally, add subtitles and customize their appearance.
4. Process and save the highlights.

## Configuration

The `ProcessingConfig` dataclass defines the default processing parameters:

```python
ProcessingConfig(
    download=False,
    language="en",
    num_highlights=10,
    min_length=15,
    max_length=40,
    add_subtitles=False,
    subtitles_language="en",
    subtitle_position="bottom",
    use_additional_video=True
)
```

## Dependencies

faster_whisper==1.0.1
ffmpeg==1.4
ffmpeg_python==0.2.0
moviepy==1.0.3
openai_whisper==20240930
opencv_contrib_python==4.10.0.84
opencv_python==4.7.0.72
opencv_python_headless==4.9.0.80
python-dotenv==1.0.1
pytubefix==8.9.0
Requests==2.32.3

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.