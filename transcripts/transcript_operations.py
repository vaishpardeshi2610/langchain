from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import sent_tokenize

def fetch_transcript(video_id):
    """
    Fetches the transcript of a YouTube video.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, languages=['en', 'hi', 'ar', 'as', 'bn', 'bg', 'zh-Hans', 'zh-Hant']
        )
        return " ".join([item['text'] for item in transcript])
    except Exception as e:
        print("Transcript not available:", e)
        return None

def split_into_sentences(text):
    """
    Splits a text into sentences using NLTK's tokenizer.
    """
    return sent_tokenize(text)
