from __future__ import annotations

"""Human-readable builtin voice list extracted from model manifest."""

from typing import Final, TypedDict


class BuiltinVoice(TypedDict):
    voice: str
    display_name: str
    group: str
    audio_file: str
    audio_file_exists: bool


BUILTIN_VOICES: Final[list[BuiltinVoice]] = [
    {
        "voice": "Junhao",
        "display_name": "CN 欢迎关注模思智能",
        "group": "Chinese Male",
        "audio_file": "zh_1.wav",
        "audio_file_exists": True,
    },
    {
        "voice": "Zhiming",
        "display_name": "CN 京味胡同闲聊",
        "group": "Chinese Male",
        "audio_file": "zh_3.wav",
        "audio_file_exists": True,
    },
    {
        "voice": "Weiguo",
        "display_name": "CN 说书",
        "group": "Chinese Male",
        "audio_file": "zh_10.wav",
        "audio_file_exists": True,
    },
    {
        "voice": "Xiaoyu",
        "display_name": "CN 明星",
        "group": "Chinese Female",
        "audio_file": "zh_11.wav",
        "audio_file_exists": True,
    },
    {
        "voice": "Yuewen",
        "display_name": "CN 机车",
        "group": "Chinese Female",
        "audio_file": "zh_4.wav",
        "audio_file_exists": True,
    },
    {
        "voice": "Lingyu",
        "display_name": "CN 深夜电台",
        "group": "Chinese Female",
        "audio_file": "zh_6.wav",
        "audio_file_exists": True,
    },
    {
        "voice": "Trump",
        "display_name": "EN Trump",
        "group": "English Male",
        "audio_file": "en_1.wav",
        "audio_file_exists": False,
    },
    {
        "voice": "Ava",
        "display_name": "EN The Bitter Lesson",
        "group": "English Female",
        "audio_file": "en_2.wav",
        "audio_file_exists": True,
    },
    {
        "voice": "Bella",
        "display_name": "EN A Gentle Reminder",
        "group": "English Female",
        "audio_file": "en_3.wav",
        "audio_file_exists": True,
    },
    {
        "voice": "Adam",
        "display_name": "EN English News",
        "group": "English Male",
        "audio_file": "en_4.wav",
        "audio_file_exists": True,
    },
    {
        "voice": "Nathan",
        "display_name": "EN The Quiet Motion of the World",
        "group": "English Male",
        "audio_file": "en_8.wav",
        "audio_file_exists": True,
    },
    {
        "voice": "Soyo",
        "display_name": "JP Soyo",
        "group": "Japanese Female",
        "audio_file": "jp_1.wav",
        "audio_file_exists": False,
    },
    {
        "voice": "Saki",
        "display_name": "JP Saki",
        "group": "Japanese Female",
        "audio_file": "jp_2.wav",
        "audio_file_exists": True,
    },
    {
        "voice": "Mortis",
        "display_name": "JP Mortis",
        "group": "Japanese Female",
        "audio_file": "jp_3.wav",
        "audio_file_exists": False,
    },
    {
        "voice": "Umiri",
        "display_name": "JP Umiri",
        "group": "Japanese Female",
        "audio_file": "jp_4.wav",
        "audio_file_exists": False,
    },
    {
        "voice": "Mei",
        "display_name": "JP Togawa",
        "group": "Japanese Female",
        "audio_file": "jp_5.wav",
        "audio_file_exists": False,
    },
    {
        "voice": "Anon",
        "display_name": "JP Anon",
        "group": "Japanese Female",
        "audio_file": "jp_6.wav",
        "audio_file_exists": False,
    },
    {
        "voice": "Arisa",
        "display_name": "JP Arisa",
        "group": "Japanese Female",
        "audio_file": "jp_7.wav",
        "audio_file_exists": False,
    },
]

BUILTIN_VOICE_MAP: Final[dict[str, BuiltinVoice]] = {
    item["voice"]: item for item in BUILTIN_VOICES
}


__all__ = ["BUILTIN_VOICES", "BUILTIN_VOICE_MAP"]
