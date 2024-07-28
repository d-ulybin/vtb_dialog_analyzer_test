import os
import librosa
import logging
from yt_dlp import YoutubeDL
import soundfile as sf

class YouTubeAudioDownloader:
    def __init__(self, output_dir="downloads", target_sr=16000):
        self.output_dir = os.path.join(os.getcwd(), output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.target_sr = target_sr

    def download_and_convert(self, youtube_url):
        video_id = self._extract_video_id(youtube_url)
        final_audio_path = os.path.join(self.output_dir, f"{video_id}_mono_{self.target_sr}hz.wav")
        
        if os.path.exists(final_audio_path):
            self.logger.info(f"Audio file already exists: {final_audio_path}")
            return final_audio_path

        temp_audio_path = os.path.join(self.output_dir, f"{video_id}.wav")

        try:
            downloaded_path = self._download_audio(youtube_url, temp_audio_path)
            self._convert_to_mono(downloaded_path, final_audio_path)
        except Exception as e:
            self.logger.error(f"Error processing {youtube_url}: {str(e)}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if os.path.exists(final_audio_path):
                os.remove(final_audio_path)
            raise
        finally:
            if os.path.exists(downloaded_path):
                os.remove(downloaded_path)

        self.logger.info(f"Mono audio saved to: {final_audio_path}")
        return final_audio_path

    def _extract_video_id(self, youtube_url):
        if "youtu.be" in youtube_url:
            return youtube_url.split("/")[-1]
        elif "youtube.com" in youtube_url:
            return youtube_url.split("v=")[1].split("&")[0]
        else:
            raise ValueError("Invalid YouTube URL")

    def _download_audio(self, youtube_url, output_path):
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        self.logger.info(f"Downloading audio from: {youtube_url}")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        
        if os.path.exists(output_path + '.wav'):
            return output_path + '.wav'
        elif os.path.exists(output_path):
            return output_path
        else:
            raise FileNotFoundError(f"Downloaded audio file not found: {output_path}")

    def _convert_to_mono(self, input_file, output_file):
        self.logger.info(f"Converting to mono and resampling to {self.target_sr} Hz: {input_file}")
        
        y, sr = librosa.load(input_file, sr=None, mono=False)
        
        if y.ndim > 1:
            y = y.mean(axis=0)
        
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
        
        sf.write(output_file, y, self.target_sr, subtype='PCM_16')