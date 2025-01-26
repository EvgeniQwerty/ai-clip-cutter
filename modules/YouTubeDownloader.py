from typing import Optional, Union, Tuple
from pathlib import Path
import ffmpeg
import re
from pytubefix import YouTube

class YouTubeDownloader:
    """Class for downloading YouTube videos with best quality audio."""
    
    def __init__(self, output_dir: Union[str, Path] = None):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory for saving videos. Default is current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def _get_size_mb(self, stream) -> float:
        """
        Get stream size in megabytes.
        
        Args:
            stream: YouTube stream object
            
        Returns:
            float: Size in MB
        """
        return stream.filesize / (1024 * 1024)

    def _sanitize_filename(self, filename: str) -> str:
        """
        Remove invalid characters from filename.
        
        Args:
            filename: Original filename
            
        Returns:
            str: Sanitized filename
        """
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Remove or replace other problematic characters
        filename = filename.replace('\'', '').replace('`', '').strip()
        return filename

    def _get_best_audio_stream(self, yt: YouTube) -> dict:
        """
        Get the best quality audio stream.
        
        Args:
            yt: YouTube object
            
        Returns:
            dict: Audio stream information
        """
        audio_streams = yt.streams.filter(only_audio=True, mime_type="audio/mp4")
        best_audio = max(audio_streams, key=lambda s: int(s.abr[:-4]))  # Remove 'kbps' and convert to int
        return {
            "stream": best_audio,
            "size_mb": self._get_size_mb(best_audio),
            "bitrate": best_audio.abr
        }

    def _get_video_streams(self, yt: YouTube) -> list[dict]:
        """
        Get available video streams (adaptive only).
        
        Args:
            yt: YouTube object
            
        Returns:
            List of video stream information
        """
        available_streams = []
        
        for stream in yt.streams.filter(type="video", adaptive=True).order_by('resolution').desc():
            if stream.resolution:
                available_streams.append({
                    "resolution": stream.resolution,
                    "stream": stream,
                    "size_mb": self._get_size_mb(stream),
                    "codec": stream.codecs[0]  # First codec is video codec
                })
                
        return available_streams

    def _merge_audio_video(self, video_path: Path, audio_path: Path, output_path: Path):
        """
        Merge video and audio files using ffmpeg.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Path to save merged file
        """
        try:
            stream = ffmpeg.input(str(video_path))
            audio = ffmpeg.input(str(audio_path))
            stream = ffmpeg.output(stream, audio, str(output_path), c='copy')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            raise ValueError(f"Error merging files: {e.stderr.decode() if e.stderr else str(e)}")

    def download_video(self, url: str, resolution: str = None) -> Path:
        """
        Download video from YouTube with best quality audio.
        
        Args:
            url: YouTube video URL
            resolution: Desired video resolution (e.g., "1080p", "720p")
                       If not specified, will prompt for choice
        
        Returns:
            Path: Absolute path to downloaded file
            
        Raises:
            ValueError: If error occurs during download or video processing
        """
        try:
            yt = YouTube(url)
            
            # Get best audio stream
            audio_info = self._get_best_audio_stream(yt)
            print(f"Selected audio: {audio_info['bitrate']} ({audio_info['size_mb']:.2f} MB)")
            
            # Get video stream
            if resolution is None:
                # Show available streams
                streams_info = self._get_video_streams(yt)
                print("\nAvailable video streams:")
                for i, stream in enumerate(streams_info):
                    print(f"{i}. Resolution: {stream['resolution']}, "
                          f"Size: {stream['size_mb']:.2f} MB, "
                          f"Codec: {stream['codec']}")
                
                while True:
                    try:
                        choice = int(input("Enter the number of the video stream to download: "))
                        if 0 <= choice < len(streams_info):
                            selected_stream = streams_info[choice]["stream"]
                            break
                        print("Invalid choice. Try again.")
                    except ValueError:
                        print("Please enter a number.")
            else:
                # Add 'p' to resolution if not present
                if not resolution.endswith('p'):
                    resolution = f"{resolution}p"
                
                streams_info = self._get_video_streams(yt)
                stream_info = next((s for s in streams_info if s["resolution"] == resolution), None)
                if not stream_info:
                    raise ValueError(f"Resolution {resolution} is not available for this video")
                selected_stream = stream_info["stream"]
            
            # Prepare filenames
            safe_title = self._sanitize_filename(yt.title)
            temp_video_path = self.output_dir / f"{safe_title}_video_temp.mp4"
            temp_audio_path = self.output_dir / f"{safe_title}_audio_temp.mp4"
            final_path = self.output_dir / f"{safe_title}.mp4"
            
            try:
                # Download video and audio
                print(f"\nDownloading video in {selected_stream.resolution}...")
                selected_stream.download(output_path=str(temp_video_path.parent),
                                      filename=temp_video_path.name)
                
                print("Downloading audio...")
                audio_info["stream"].download(output_path=str(temp_audio_path.parent),
                                           filename=temp_audio_path.name)
                
                # Merge files
                print("Merging audio and video...")
                self._merge_audio_video(temp_video_path, temp_audio_path, final_path)
                
                return final_path.absolute()
                
            finally:
                # Clean up temporary files
                if temp_video_path.exists():
                    temp_video_path.unlink()
                if temp_audio_path.exists():
                    temp_audio_path.unlink()
            
        except Exception as e:
            raise ValueError(f"Error downloading video: {str(e)}")

if __name__ == "__main__":
    videostream = YouTubeDownloader(f"{Path.cwd()}/videos")            
    videostream.download_video("https://youtu.be/WptcQ1cyrUc")