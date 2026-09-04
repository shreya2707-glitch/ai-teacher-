"""
AI Teacher - Avatar & Video Generation
Turns the `spoken_script` from lesson_engine.explain_concept() into an actual
talking-head teaching video via D-ID's API.

Two modes, pick based on how much multilingual quality you need:

MODE A - "simple" (fastest to wire up):
    Send text straight to D-ID, let D-ID's built-in TTS + avatar handle everything
    in one call. Good enough for English/Hindi if D-ID's voice list covers your
    language well - check their /voices list before demo day.

MODE B - "custom_audio" (better multilingual control):
    Generate audio yourself first (Google Cloud TTS / ElevenLabs - both have much
    stronger Hindi/Hinglish voices), upload/host that audio, then pass the audio
    URL to D-ID instead of text. More moving parts, better quality.

Requires:
    pip install requests --break-system-packages
    D-ID account + API key: https://www.d-id.com/api/
    Set D_ID_API_KEY as an environment variable.

D-ID docs: https://docs.d-id.com/reference/createtalk
"""

import os
import time
import requests

D_ID_API_KEY = os.environ.get("D_ID_API_KEY")  # format: "username:password" base64'd, or bearer token per your D-ID plan
D_ID_BASE_URL = "https://api.d-id.com"

# A presenter/avatar image D-ID can animate. Swap for your own hosted image URL -
# use a clean front-facing photo/illustration, D-ID's docs list requirements.
DEFAULT_PRESENTER_IMAGE_URL = os.environ.get(
    "PRESENTER_IMAGE_URL",
    "https://create-images-results.d-id.com/DefaultPresenters/Emma_f/image.jpeg",
)

HEADERS = {
    "Authorization": f"Basic {D_ID_API_KEY}",
    "Content-Type": "application/json",
}


# ---------------------------------------------------------------------------
# MODE A - text straight to D-ID (D-ID's own TTS + avatar)
# ---------------------------------------------------------------------------

# D-ID voice IDs vary by provider (microsoft/amazon/elevenlabs passthrough).
# These are common Microsoft neural voices with decent Hindi support - verify
# current options at https://docs.d-id.com/reference/createtalk before demo day.
LANGUAGE_VOICE_MAP = {
    "english": "en-US-JennyNeural",
    "hindi": "hi-IN-SwaraNeural",
    "hinglish": "hi-IN-SwaraNeural",   # closest available; code-switching handled in the script itself
}


def create_talk_simple(spoken_script: str, language: str = "english",
                        presenter_image_url: str = DEFAULT_PRESENTER_IMAGE_URL) -> str:
    """
    Submits a talk generation job to D-ID using text + D-ID's built-in TTS.
    Returns the D-ID talk `id` - poll get_talk_result() with this id.
    """
    voice_id = LANGUAGE_VOICE_MAP.get(language.lower(), LANGUAGE_VOICE_MAP["english"])

    payload = {
        "source_url": presenter_image_url,
        "script": {
            "type": "text",
            "input": spoken_script,
            "provider": {"type": "microsoft", "voice_id": voice_id},
        },
        "config": {"fluent": True, "pad_audio": 0.0},
    }

    resp = requests.post(f"{D_ID_BASE_URL}/talks", headers=HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# MODE B - pre-generated audio (better multilingual quality) to D-ID
# ---------------------------------------------------------------------------

def create_talk_from_audio(audio_url: str,
                            presenter_image_url: str = DEFAULT_PRESENTER_IMAGE_URL) -> str:
    """
    Use this when you've already generated TTS audio elsewhere (e.g. ElevenLabs
    multilingual, Google Cloud TTS) and hosted it somewhere D-ID can fetch it
    (a public URL - S3, Supabase storage, even a temporary file host).
    Returns the D-ID talk id.
    """
    payload = {
        "source_url": presenter_image_url,
        "script": {"type": "audio", "audio_url": audio_url},
        "config": {"fluent": True, "pad_audio": 0.0},
    }
    resp = requests.post(f"{D_ID_BASE_URL}/talks", headers=HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Poll for completion - D-ID rendering is async, typically 10-40s for short clips
# ---------------------------------------------------------------------------

def get_talk_result(talk_id: str, poll_interval_sec: float = 2.0, timeout_sec: float = 90.0) -> dict:
    """
    Polls D-ID until the talk is done or fails. Returns the full talk object;
    the video is at result["result_url"] once result["status"] == "done".
    """
    elapsed = 0.0
    while elapsed < timeout_sec:
        resp = requests.get(f"{D_ID_BASE_URL}/talks/{talk_id}", headers=HEADERS)
        resp.raise_for_status()
        result = resp.json()

        if result["status"] == "done":
            return result
        if result["status"] == "error":
            raise RuntimeError(f"D-ID talk generation failed: {result}")

        time.sleep(poll_interval_sec)
        elapsed += poll_interval_sec

    raise TimeoutError(f"D-ID talk {talk_id} did not complete within {timeout_sec}s")


# ---------------------------------------------------------------------------
# Optional: Google Cloud TTS helper for MODE B, since its Hindi support is solid
# and it's cheap. Skip this if you go MODE A only.
# ---------------------------------------------------------------------------

def generate_tts_google(text: str, language_code: str = "hi-IN",
                         voice_name: str = "hi-IN-Wavenet-A", output_path: str = "output.mp3") -> str:
    """
    Requires: pip install google-cloud-texttospeech --break-system-packages
    and GOOGLE_APPLICATION_CREDENTIALS env var pointing to a service account JSON.
    Returns the local output_path - you still need to upload this somewhere
    publicly reachable before passing its URL to create_talk_from_audio().
    """
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    with open(output_path, "wb") as f:
        f.write(response.audio_content)
    return output_path


# ---------------------------------------------------------------------------
# End-to-end helper - wires straight onto lesson_engine.explain_concept() output
# ---------------------------------------------------------------------------

def generate_teaching_video(explanation: dict, language: str = "english",
                             presenter_image_url: str = DEFAULT_PRESENTER_IMAGE_URL) -> str:
    """
    explanation: the dict returned by lesson_engine.explain_concept()
                 (must contain "spoken_script").
    Returns the final video URL.

    This is MODE A (simplest). For MODE B, generate audio first with
    generate_tts_google() or ElevenLabs, upload it, then call
    create_talk_from_audio() + get_talk_result() instead.
    """
    talk_id = create_talk_simple(
        spoken_script=explanation["spoken_script"],
        language=language,
        presenter_image_url=presenter_image_url,
    )
    result = get_talk_result(talk_id)
    return result["result_url"]


if __name__ == "__main__":
    # smoke test - requires D_ID_API_KEY to be set
    fake_explanation = {
        "spoken_script": "Let's talk about Ohm's Law. It tells us how voltage, "
                          "current, and resistance relate to each other in a circuit."
    }
    video_url = generate_teaching_video(fake_explanation, language="english")
    print("Video ready at:", video_url)