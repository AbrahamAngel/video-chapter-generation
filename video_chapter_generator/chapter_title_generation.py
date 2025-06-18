# chapter_title_generation.py
import json
import os
from transformers import pipeline

def generate_chapter_title_simple(chapter_text, max_words=10):
    """
    Generates a simple chapter title by taking the first few words of the chapter.
    """
    words = chapter_text.split()
    title = " ".join(words[:max_words]).strip()
    if len(words) > max_words:
        title += "..."
    return title

def generate_chapter_title_summarizer(chapter_text, min_length=20, max_length=50):
    """
    Generates a chapter title using a pre-trained summarization model (e.g., T5).
    Requires 'transformers' library.
    """
    try:
        # Load the summarizer model only once if possible for performance
        if not hasattr(generate_chapter_title_summarizer, 'summarizer_pipeline'):
            print("Loading summarization model (this happens once)...")
            generate_chapter_title_summarizer.summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        
        summarizer = generate_chapter_title_summarizer.summarizer_pipeline
        
        # Summarize the first N characters to avoid overwhelming the model with very long texts
        # 2000 characters is a reasonable limit for many summarization models
        summary = summarizer(chapter_text[:2000], min_length=min_length, max_length=max_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Warning: Could not use summarization model. Falling back to simple title generation. Error: {e}")
        return generate_chapter_title_simple(chapter_text, max_words=15)


def generate_titles_for_chapters(chapters_data):
    """
    Generates titles for a list of chapter dictionaries.
    """
    print("Generating chapter titles...")
    for i, chapter in enumerate(chapters_data):
        chapter['title'] = generate_chapter_title_summarizer(chapter['text'])
        print(f"Generated title for Chapter {i+1}: '{chapter['title']}'")
    return chapters_data

if __name__ == "__main__":
    # Example Usage: (This block is primarily for testing this module in isolation)
    chapters_file = "raw_chapters.json" # Placeholder
    if os.path.exists(chapters_file):
        with open(chapters_file, "r", encoding="utf-8") as f:
            raw_chapters = json.load(f)

        if raw_chapters:
            titled_chapters = generate_titles_for_chapters(raw_chapters)
            print(f"\nTitles generated for {len(titled_chapters)} chapters (if data available).")
        else:
            print("No raw chapters loaded.")
    else:
        print(f"Reminder: To run this module standalone, ensure '{chapters_file}' exists or update path.")