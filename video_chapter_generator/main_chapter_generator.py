# main_chapter_generator.py
import os
import json
from datetime import timedelta

# Import functions from previous modules
from video_to_audio import extract_audio
from audio_to_text import transcribe_audio
from topic_segmentation import get_sentences_with_timestamps, identify_topic_boundaries, group_segments_into_chapters
from chapter_title_generation import generate_titles_for_chapters

def format_timestamp(seconds):
    """Formats seconds into HH:MM:SS string."""
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def generate_video_chapters(video_path, whisper_model_name="base", similarity_threshold=0.6):
    """
    Generates chapter markers for a given video file.

    Args:
        video_path (str): Path to the input video file.
        whisper_model_name (str): The Whisper model to use (e.g., 'base', 'small', 'medium').
        similarity_threshold (float): Threshold for topic segmentation (0.0 to 1.0).

    Returns:
        list: A list of dictionaries, each representing a chapter with 'timestamp' and 'title'.
              Returns None if the process fails.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return None

    video_filename = os.path.basename(video_path)
    video_basename_no_ext = os.path.splitext(video_filename)[0]

    # Define paths for intermediate and final output files based on video name
    audio_output_path = f"{video_basename_no_ext}_extracted_audio.mp3"
    transcript_segments_path = f"{video_basename_no_ext}_transcript_segments.json"
    raw_chapters_path = f"{video_basename_no_ext}_raw_chapters.json"
    final_chapters_path = f"{video_basename_no_ext}_final_chapters.json"

    # --- Step 1: Extract Audio ---
    audio_file = extract_audio(video_path, audio_output_path)
    if not audio_file:
        return None

    # --- Step 2: Transcribe Audio ---
    segments_data, _ = transcribe_audio(audio_file, model_name=whisper_model_name)
    if not segments_data:
        os.remove(audio_file) # Clean up audio if transcription failed
        return None
    with open(transcript_segments_path, "w", encoding="utf-8") as f:
        json.dump(segments_data, f, ensure_ascii=False, indent=4)
    print(f"Transcript segments saved to {transcript_segments_path}")

    # --- Step 3 & 4: Topic Segmentation ---
    sentences_with_ts = get_sentences_with_timestamps(segments_data)
    if not sentences_with_ts:
        print("No sentences extracted for topic segmentation. Aborting.")
        os.remove(audio_file)
        os.remove(transcript_segments_path)
        return None

    boundary_indices, _ = identify_topic_boundaries(sentences_with_ts, threshold=similarity_threshold)
    raw_chapters = group_segments_into_chapters(segments_data, boundary_indices, sentences_with_ts)
    if not raw_chapters:
        print("No raw chapters generated. Aborting.")
        os.remove(audio_file)
        os.remove(transcript_segments_path)
        return None
    with open(raw_chapters_path, "w", encoding="utf-8") as f:
        json.dump(raw_chapters, f, ensure_ascii=False, indent=4)
    print(f"Raw chapters saved to {raw_chapters_path}")

    # --- Step 5: Generate Chapter Titles ---
    final_chapters = generate_titles_for_chapters(raw_chapters)
    with open(final_chapters_path, "w", encoding="utf-8") as f:
        json.dump(final_chapters, f, ensure_ascii=False, indent=4)
    print(f"Final chapters saved to {final_chapters_path}")

    # --- Step 6: Format Output ---
    chapter_markers = []
    for chapter in final_chapters:
        chapter_markers.append({
            "timestamp": format_timestamp(chapter['start_time']),
            "title": chapter['title']
        })

    # Optional cleanup: uncomment these lines if you want to remove intermediate files
    # print("\nCleaning up intermediate files...")
    # os.remove(audio_file)
    # os.remove(transcript_segments_path)
    # os.remove(raw_chapters_path)

    print("\n--- Generated Chapter Markers ---")
    for marker in chapter_markers:
        print(f"{marker['timestamp']} {marker['title']}")

    return chapter_markers

if __name__ == "__main__":
    # --- IMPORTANT: Configure your video path and parameters here ---
    input_video = "sample.mp4" # <--- REPLACE THIS with the actual path to YOUR video file
                                     # Example: "C:/Users/YourUser/Videos/my_long_lecture.mp4"
                                     # Example: "/home/user/videos/tutorial.mkv"
    
    WHISPER_MODEL = "base" # Options: 'tiny', 'base', 'small', 'medium', 'large'
                           # 'base' is a good starting point for CPU. 'small' might be too slow.
    
    SIMILARITY_THRESHOLD = 0.65 # Adjust this value (e.g., 0.5 to 0.8) to control chapter granularity.
                                # Lower value = more chapters (more sensitive to topic changes).
                                # Higher value = fewer chapters (less sensitive).

    chapters = generate_video_chapters(input_video, WHISPER_MODEL, SIMILARITY_THRESHOLD)

    if chapters:
        print("\nChapter generation completed successfully!")
    else:
        print("\nChapter generation failed.")