# topic_segmentation.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import nltk
from nltk.tokenize import sent_tokenize
import os

# Download NLTK punkt tokenizer for sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def get_sentences_with_timestamps(segments_data):
    """
    Extracts sentences from Whisper segments, preserving start timestamps.
    Whisper segments are often phrase-level, so we re-segment to full sentences.
    """
    sentences_with_ts = []
    current_text = ""
    current_start_time = 0.0

    if not segments_data:
        return []

    for i, segment in enumerate(segments_data):
        if i == 0:
            current_start_time = segment['start']

        current_text += " " + segment['text']

        if segment['text'].strip().endswith(('.', '?', '!')) or i == len(segments_data) - 1:
            sentences = sent_tokenize(current_text.strip())
            for j, sentence in enumerate(sentences):
                if j == 0:
                    sentences_with_ts.append({'text': sentence, 'start': current_start_time})
                else:
                    sentences_with_ts.append({'text': sentence, 'start': segments_data[i]['end'] - len(sentence.split()) * 0.3}) # Estimate
            current_text = ""
            if i < len(segments_data) - 1:
                current_start_time = segments_data[i+1]['start']

    return sentences_with_ts


def identify_topic_boundaries(sentences_with_timestamps, threshold=0.6):
    """
    Identifies topic boundaries in a list of sentences using semantic similarity.

    Args:
        sentences_with_timestamps (list): List of dictionaries, each with 'text' and 'start'.
        threshold (float): Similarity threshold. A dip below this might indicate a topic change.

    Returns:
        list: A list of indices where new topics likely begin (sentence index).
    """
    if not sentences_with_timestamps:
        return [], []

    print("Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Generating sentence embeddings...")
    sentences = [s['text'] for s in sentences_with_timestamps]
    embeddings = model.encode(sentences)

    print("Calculating cosine similarities between adjacent sentences...")
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        similarities.append(sim)

    boundaries = [0] # The first sentence is always the start of the first chapter

    for i, sim in enumerate(similarities):
        if sim < threshold:
            boundaries.append(i + 1)

    boundaries = sorted(list(set(boundaries)))
    print(f"Identified {len(boundaries)} potential topic boundaries.")
    return boundaries, similarities


def group_segments_into_chapters(segments_data, topic_boundary_indices, sentences_with_timestamps):
    """
    Groups the original Whisper segments into chapters based on topic boundaries.
    """
    chapters = []
    current_chapter_segments = []
    current_chapter_start_time = 0.0
    current_chapter_title_text = []

    sentence_start_times = [s['start'] for s in sentences_with_timestamps]

    segment_idx = 0

    if not segments_data:
        return []

    for i, boundary_idx in enumerate(topic_boundary_indices):
        chapter_start_time = sentence_start_times[boundary_idx]

        next_boundary_time = float('inf')
        if (i + 1) < len(topic_boundary_indices):
            next_boundary_time = sentence_start_times[topic_boundary_indices[i + 1]]
        elif (i + 1) == len(topic_boundary_indices) and len(topic_boundary_indices) > 0:
             next_boundary_time = segments_data[-1]['end'] + 1


        chapter_segments = []
        while segment_idx < len(segments_data) and segments_data[segment_idx]['start'] < next_boundary_time:
            chapter_segments.append(segments_data[segment_idx])
            segment_idx += 1

        if chapter_segments:
            chapter_text = " ".join([s['text'] for s in chapter_segments])
            chapters.append({
                'start_time': chapter_segments[0]['start'],
                'end_time': chapter_segments[-1]['end'],
                'text': chapter_text,
                'title': ""
            })

    return chapters

if __name__ == "__main__":
    # Example Usage: (This block is primarily for testing this module in isolation)
    segments_file = "transcript_segments.json" # Placeholder
    if os.path.exists(segments_file):
        with open(segments_file, "r", encoding="utf-8") as f:
            segments_data = json.load(f)

        if segments_data:
            sentences_with_ts = get_sentences_with_timestamps(segments_data)
            if sentences_with_ts:
                print(f"Processing {len(sentences_with_ts)} sentences for topic segmentation.")
                boundary_indices, similarities = identify_topic_boundaries(sentences_with_ts, threshold=0.6)
                chapters = group_segments_into_chapters(segments_data, boundary_indices, sentences_with_ts)
                print(f"\nCreated {len(chapters)} raw chapters (if data available).")
            else:
                print("No sentences extracted.")
        else:
            print("No segments data loaded.")
    else:
        print(f"Reminder: To run this module standalone, ensure '{segments_file}' exists or update path.")