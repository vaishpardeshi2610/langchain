from youtube_transcript_api import YouTubeTranscriptApi

def fetch_and_save_transcript(video_id, output_file="transcript.txt", languages=None):
    """Fetches YouTube transcript and saves to a file."""
    languages = languages or ['en']
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)

    # Combine short entries for better context
    combined_transcript = []
    current_chunk = ""
    for entry in transcript:
        if len(current_chunk) + len(entry['text']) < 500:  # Combine entries up to 500 characters
            current_chunk += " " + entry['text']
        else:
            combined_transcript.append(current_chunk.strip())
            current_chunk = entry['text']
    if current_chunk:  # Add the last chunk
        combined_transcript.append(current_chunk.strip())

    with open(output_file, "w") as file:
        for chunk in combined_transcript:
            file.write(chunk + '\n')

    return output_file
