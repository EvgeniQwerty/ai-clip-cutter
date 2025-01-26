import cv2
import whisper
from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip
from moviepy.config import change_settings
from datetime import datetime
import os
import random
from pathlib import Path
from typing import List, Dict, Literal
import json

class SubtitlePosition:
    """Constants for subtitle positioning"""
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"

    # Safe padding for TikTok interface (in percentage of video height)
    TIKTOK_TOP_PADDING = 0.15  # 15% from top for avatar and description
    TIKTOK_BOTTOM_PADDING = 0.2  # 20% from bottom for buttons, description and music
    TIKTOK_CENTER_PADDING = 0.1  # 10% from center for safe zone

class SubtitleProcessor:
    def __init__(self):
        self.model = whisper.load_model("base")
    
    def _split_subtitle(self, subtitle: Dict) -> List[Dict]:
        """
        Splits a subtitle into chunks of maximum 3 words.
        
        Args:
            subtitle (Dict): Original subtitle with 'start', 'end', and 'text'
            
        Returns:
            List[Dict]: List of shorter subtitles
        """
        words = subtitle['text'].split()
        chunks = []
        duration = subtitle['end'] - subtitle['start']
        
        # Split words into chunks of max 3 words
        for i in range(0, len(words), 3):
            chunk = words[i:i + 3]
            chunk_text = ' '.join(chunk)
            
            # Calculate time proportions for each chunk
            chunk_duration = duration * (len(chunk) / len(words))
            chunk_start = subtitle['start'] + (duration * (i / len(words)))
            chunk_end = chunk_start + chunk_duration
            
            chunks.append({
                'start': chunk_start,
                'end': chunk_end,
                'text': chunk_text
            })
        
        return chunks
        
    def extract_subtitles(self, video_path: str, language: str) -> List[Dict]:
        """
        Extracts text from video audio track and splits into short phrases.
        
        Args:
            video_path (str): Path to the video file
            language (str): Language code for transcription
            
        Returns:
            List[Dict]: List of dictionaries with recognized text and timestamps
        """
        result = self.model.transcribe(video_path, language=language)
        
        # Get original subtitles
        original_subtitles = [
            {
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            }
            for segment in result['segments']
        ]
        
        # Split each subtitle into shorter chunks
        short_subtitles = []
        for subtitle in original_subtitles:
            short_subtitles.extend(self._split_subtitle(subtitle))
            
        return short_subtitles

class VideoProcessor:
    def __init__(self):
        change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.subtitle_processor = SubtitleProcessor()
        
    def _calculate_subtitle_position(self, video_height: int, position: str) -> tuple:
        """
        Calculates safe subtitle position considering TikTok interface.
        
        Args:
            video_height (int): Video height in pixels
            position (str): Desired subtitle position (top/center/bottom)
            
        Returns:
            tuple: (x, y) coordinates for subtitle placement
        """
        if position == SubtitlePosition.TOP:
            # Place below the top safe zone
            y_pos = int(video_height * SubtitlePosition.TIKTOK_TOP_PADDING)
        elif position == SubtitlePosition.CENTER:
            # Place in center with slight upward offset
            center = video_height // 2
            y_pos = int(center - (center * SubtitlePosition.TIKTOK_CENTER_PADDING))
        else:  # BOTTOM
            # Place above the bottom safe zone
            y_pos = int(video_height * (1 - SubtitlePosition.TIKTOK_BOTTOM_PADDING))
            
        return ('center', y_pos)
        
    def add_subtitles(self, video: VideoFileClip, subtitles: List[Dict], 
                      position: Literal["top", "center", "bottom"] = "bottom") -> VideoFileClip:
        """
        Adds subtitles to video.
        
        Args:
            video (VideoFileClip): Video clip
            subtitles (List[Dict]): List of subtitles with timestamps
            position (str): Subtitle position (top/center/bottom)
            
        Returns:
            VideoFileClip: Video with subtitles
        """
        subtitle_clips = []
        video_height = video.size[1]
        
        # Calculate font size based on video height
        fontsize = int(video_height * 0.06)  # 6% of video height
        
        # Define maximum text width (70% of video width)
        max_width = int(video.size[0] * 0.7)
        
        subtitle_position = self._calculate_subtitle_position(video_height, position)
        
        for subtitle in subtitles:
            # Create text clip with width limitation
            text_clip = (TextClip(subtitle['text'], 
                                font=f'{Path.cwd()}/Roboto-Medium.ttf', 
                                fontsize=fontsize,
                                color='white',
                                stroke_color='black',
                                stroke_width=2,
                                size=(max_width, None),
                                method='caption')
                        .set_position(subtitle_position)
                        .set_start(subtitle['start'])
                        .set_end(subtitle['end']))
            subtitle_clips.append(text_clip)
            
        return CompositeVideoClip([video] + subtitle_clips)
    
    def process_video(self, input_path: str, start_time: float, end_time: float,
                  language: str = "en", use_additional_video: bool = False,
                  subtitle_position: Literal["top", "center", "bottom"] = "bottom",
                  add_subtitles: bool = True) -> str:
        """
        Processes video and optionally adds subtitles.

        Args:
            input_path (str): Path to input video
            start_time (float): Start time for trimming
            end_time (float): End time for trimming
            language (str): Language code for transcription
            use_additional_video (bool): Whether to use additional video
            subtitle_position (str): Subtitle position (top/center/bottom)
            add_subtitles (bool): Flag to enable or disable subtitle addition

        Returns:
            str: Path to output video
        """
        main_video = VideoFileClip(input_path).subclip(start_time, end_time)
        clip_duration = end_time - start_time
    
        subtitles = []
        if add_subtitles:
            # Recognize speech and prepare subtitles
            subtitles = self.subtitle_processor.extract_subtitles(input_path, language)
            trimmed_subtitles = [
                subtitle for subtitle in subtitles
                if subtitle['start'] >= start_time and subtitle['end'] <= end_time
            ]
            for subtitle in trimmed_subtitles:
                subtitle['start'] -= start_time
                subtitle['end'] -= start_time

        if use_additional_video:
            main_processed = self._process_main_video_split(main_video)
            additional_processed = self._get_random_additional_video(clip_duration, main_processed.size)
            final_video = clips_array([[additional_processed], [main_processed]])
        else:
            main_processed = self._process_main_video_full(main_video)
            final_video = main_processed

        # Add subtitles if enabled
        if add_subtitles:
            final_video = self.add_subtitles(
                final_video,
                trimmed_subtitles,
                position=subtitle_position
                )

        output_filename = self._generate_output_filename(input_path)
        final_video.write_videofile(output_filename)

        if add_subtitles:
            # Save subtitles separately in JSON format
            subtitles_filename = output_filename.rsplit('.', 1)[0] + '_subtitles.json'
            with open(subtitles_filename, 'w', encoding='utf-8') as f:
                json.dump(trimmed_subtitles, f, ensure_ascii=False, indent=2)

        return output_filename
    
    def _process_main_video_full(self, video):
        """
        Crops the main video for full-screen layout (aspect ratio 9:16).
        
        Args:
            video (VideoFileClip): The video to process.
        
        Returns:
            VideoFileClip: Cropped video.
        """
        target_aspect_ratio = 9 / 16
        width, height = video.size
        return self._crop_to_aspect_ratio(video, target_aspect_ratio, width, height)

    def _process_main_video_split(self, video):
        """
        Crops the main video for split-screen layout (aspect ratio 9:8).
        
        Args:
            video (VideoFileClip): The video to process.
        
        Returns:
            VideoFileClip: Cropped video.
        """
        target_aspect_ratio = 9 / 8
        width, height = video.size
        return self._crop_to_aspect_ratio(video, target_aspect_ratio, width, height)
    
    def _get_random_additional_video(self, duration, target_size):
        """
        Retrieves and processes a random additional video to match the target size.
        
        Args:
            duration (float): Duration of the additional video in seconds.
            target_size (tuple): Target size (width, height) for resizing.
        
        Returns:
            VideoFileClip: Processed additional video.
        """
        additional_videos_dir = "additional_videos"
        video_files = [f for f in os.listdir(additional_videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        random_video_path = os.path.join(additional_videos_dir, random.choice(video_files))
        additional_video = VideoFileClip(random_video_path)
        
        max_start = max(0, additional_video.duration - duration)
        random_start = random.uniform(0, max_start)
        additional_video = additional_video.subclip(random_start, random_start + duration)

        # Remove audio from the additional video
        additional_video = additional_video.without_audio()
    
        target_aspect_ratio = 9 / 8
        width, height = additional_video.size
        cropped_video = self._crop_to_aspect_ratio(additional_video, target_aspect_ratio, width, height)
        
        # Resize additional video to match the target size
        return cropped_video.resize(target_size)
    
    def _crop_to_aspect_ratio(self, video, target_aspect_ratio, width, height):
        """
        Crops a video to match the specified aspect ratio.
        
        Args:
            video (VideoFileClip): The video to crop.
            target_aspect_ratio (float): Desired aspect ratio (width/height).
            width (int): Original width of the video.
            height (int): Original height of the video.
        
        Returns:
            VideoFileClip: Cropped video.
        """
        input_aspect_ratio = width / height

        if input_aspect_ratio > target_aspect_ratio:
            # Crop width to match the target aspect ratio
            target_width = int(height * target_aspect_ratio)
            crop_x = (width - target_width) // 2
            return video.crop(x1=crop_x, y1=0, x2=crop_x + target_width, y2=height)
        elif input_aspect_ratio < target_aspect_ratio:
            # Crop height to match the target aspect ratio
            target_height = int(width / target_aspect_ratio)
            crop_y = (height - target_height) // 2
            return video.crop(x1=0, y1=crop_y, x2=width, y2=crop_y + target_height)
        return video
    
    def _get_face_position(self, frame, target_width):
        """
        Determines the cropping position based on detected faces or centers the crop.
        
        Args:
            frame (numpy.ndarray): A single frame from the video.
            target_width (int): Desired width of the crop.
        
        Returns:
            int: X-coordinate for cropping.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, _, w, _ = faces[0]
            center_x = x + w / 2
            return int(center_x - target_width / 2)
        return (frame.shape[1] - target_width) // 2
    
    def _generate_output_filename(self, input_path):
        """
        Generates a filename for saving the processed video.
        
        Args:
            input_path (str): Path to the input video file.
        
        Returns:
            str: Generated output file path.
        """
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("output", exist_ok=True)
        return f"output/{base_name}_highlight_{current_datetime}.mp4"

if __name__ == "__main__":
    processor = VideoProcessor()
    print("Processing video...")
    output_path = processor.process_video(
        input_path=f"{Path.cwd()}/videos/video.mp4",
        start_time=295.62,
        end_time=335.62,
        language="ru",
        use_additional_video=True,
        subtitle_position="bottom",  # Use "top", "center" or "bottom"
        add_subtitles=False
    )
    print(f"Video processing complete. Output saved to: {output_path}")