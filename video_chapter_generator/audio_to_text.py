# audio_to_text.py
import whisper
import json
import os

def transcribe_audio(audio_path, model_name="base"):
    """
    Transcribes an audio file using OpenAI Whisper and returns a detailed transcript
    including word-level timestamps.

    Args:
        audio_path (str): Path to the input audio file.
        model_name (str): The Whisper model to use (e.g., 'tiny', 'base', 'small', 'medium', 'large').

    Returns:
        list: A list of dictionaries, where each dictionary represents a segment
              with 'start', 'end', and 'text' (sentence/phrase level).
              Example: [{'start': 0.0, 'end': 5.2, 'text': 'Hello world'}, ...]
        list: A list of dictionaries for word-level timestamps.
              Example: [{'word': 'Hello', 'start': 0.0, 'end': 0.5}, ...]
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        return None, None

    print(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)
    print(f"Transcribing audio: {audio_path} (This may take a while for long audios)...")

    result = model.transcribe(audio_path, word_timestamps=True)

    segments = result['segments']
    word_timestamps = []

    for segment in segments:
        for word_info in segment['words']:
            word_timestamps.append({
                'word': word_info['word'].strip(),
                'start': word_info['start'],
                'end': word_info['end']
            })

    print("Transcription complete.")
    return segments, word_timestamps

if __name__ == "__main__":
    # Example Usage: (This block is primarily for testing this module in isolation)
    audio_file = "extracted_audio.mp3" # Placeholder

    if os.path.exists(audio_file):
        segments, word_ts = transcribe_audio(audio_file, model_name="base")
        if segments and word_ts:
            print("\n--- Segment-level Transcript (First 5) ---")
            for segment in segments[:5]:
                print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
            print("\nTranscripts generated.")
        else:
            print("Transcription failed.")
    else:
        print(f"Reminder: To run this module standalone, place '{audio_file}' or update path.")