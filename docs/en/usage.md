# Usage

## Main Flow

1. Open or create a project.
2. Prepare media in **Video Downloader** from an online link, a local upload, or an existing project file.
3. Run **Speech to Text** to create a transcript and subtitles.
4. Run **Text Translation** to translate the subtitles.
5. Open **Video Player** to watch the translated subtitles with the video and adjust the on-page viewing style.

## Choosing Models

Model cards in **Settings** are ordered by priority. The first card is the default model shown in the workflow pages.

The **Speech to Text** and **Text Translation** pages separate cloud models and local models so you can clearly see where the selected model runs:

- cloud models send requests to the service you configure
- local models run on your machine after the required model files are installed

If one side has no models, the selector keeps the other side available and shows an empty state for the missing group.

## Speech to Text Tips

- If you already know the audio language, selecting it directly is often more reliable than auto-detect.
- Use word alignment when you need tighter subtitle timing.
- Use VAD when the audio contains long quiet sections or when you want speech regions detected before recognition.
- Use speaker diarization when you want subtitles grouped by speaker. pyannote can be installed from **Settings** when needed.

## Translation Tips

- Check the transcript before translating, especially if names or specialized terms matter.
- Use a glossary when the same names, brands, or terms should be translated consistently.
- Use the prompt field to describe tone, names, or reading preferences that the model should follow.
- Subtitle alignment repair can help keep one translated line matched to one source subtitle line when the chosen model supports that workflow well.

## Local Models

To use local ASR or local translation:

1. Open **Settings**.
2. Choose the local ASR or local translation model area.
3. Enter a Hugging Face model id and inspect it.
4. Install the model. Large downloads can take time and continue as a background task.
5. Return to the workflow page and select the installed model.

Local models keep processing on your machine, but they still require enough disk space and suitable hardware. Cloud models are usually faster to start, but media text is sent to the configured provider.

## Video Player

**Video Player** is for watching the translated subtitle result with the video. Subtitle style controls in Video Player change the viewing experience on that page.
