# Usage

## Main Flow

1. Open or create a project
2. Import media or download it in **Video Downloader**
3. Run **Speech to Text**
4. Run **Text Translation**
5. Review and export in **Player**

## Speech to Text Tips

- If you already know the language, selecting it directly is often more stable than auto-detect
- If you want speaker diarization with pyannote, install pyannote first in **Settings**
- If pyannote is not installed yet, ArcSub will keep the classic diarization option available

## Translation Tips

- Check the transcript before translating
- Use your own glossary when names or technical terms must stay consistent
- If you only need a simple result, start with the default settings

## Local Models

If you want local ASR or local translation:

1. Open **Settings**
2. Choose **ASR Model** or **Translation Model**
3. Enter the Hugging Face model id and inspect it
4. Install the model; long downloads or conversions continue as a background task
5. Return to the workflow page and select it

ArcSub reads trusted Hugging Face metadata when available and derives model-specific local defaults such as runtime hints, chat-template support, and token-aware translation batching.

`HF_TOKEN` is shared by pyannote and gated/private Hugging Face model downloads. Public models usually do not need it.
