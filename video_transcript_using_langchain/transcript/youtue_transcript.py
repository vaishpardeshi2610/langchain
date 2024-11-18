from youtube_transcript_api import YouTubeTranscriptApi

def fetch_and_save_transcript(video_id, output_file="transcript.txt", languages=None):
    """Fetches YouTube transcript and saves to a file."""
    languages = languages or ['en']
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)

    with open(output_file, "w") as file:
        for entry in transcript:
            file.write(entry['text'] + '\n')

    return output_file
