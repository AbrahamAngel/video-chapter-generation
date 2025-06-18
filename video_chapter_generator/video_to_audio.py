# video_to_audio.py
from moviepy.editor import VideoFileClip
import os

def extract_audio(video_path, audio_output_path):
    """
    Extracts audio from a video file.

    Args:
        video_path (str): Path to the input video file.
        audio_output_path (str): Path where the extracted audio will be saved (e.g., "audio.mp3").

    Returns:
        str: Path to the extracted audio file.
    """
    try:
        print(f"Extracting audio from {video_path}...")
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_output_path)
        video_clip.close()
        audio_clip.close()
        print(f"Audio extracted to {audio_output_path}")
        return audio_output_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

if __name__ == "__main__":
    # Example Usage: (This block is primarily for testing this module in isolation)
    video_file = "sample_video.mp4" # Placeholder
    audio_file = "extracted_audio.mp3"

    if os.path.exists(video_file):
        extracted_audio_path = extract_audio(video_file, audio_file)
        if extracted_audio_path:
            print(f"Audio extraction successful: {extracted_audio_path}")
        else:
            print("Audio extraction failed.")
    else:
        print(f"Reminder: To run this module standalone, place '{video_file}' or update path.")