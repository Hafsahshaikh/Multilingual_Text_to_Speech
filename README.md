# Multilingual Text-to-Speech Using Hugging Face

A Google Colab-based multilingual Text-to-Speech (TTS) project that converts written text into speech using pretrained Hugging Face models. The project supports **English, Hindi, Telugu, and Turkish** and allows users to generate, listen to, and download speech as WAV audio files.

The notebook also includes basic audio-signal diagnostics, an optional pretrained speech-quality predictor, and manual listening ratings.

## Project Overview

Text-to-Speech technology converts written text into spoken audio. It is useful in accessibility applications, reading assistants, language-learning tools, and voice-based interfaces.

This project demonstrates how pretrained deep-learning models can be integrated into a practical multilingual speech-generation workflow using Python, PyTorch, and Hugging Face Transformers.

## Supported Languages and Models

| Language | Hugging Face Model |
|---|---|
| English | [`facebook/mms-tts-eng`](https://huggingface.co/facebook/mms-tts-eng) |
| Hindi | [`facebook/mms-tts-hin`](https://huggingface.co/facebook/mms-tts-hin) |
| Telugu | [`facebook/mms-tts-tel`](https://huggingface.co/facebook/mms-tts-tel) |
| Turkish | [`facebook/mms-tts-tur`](https://huggingface.co/facebook/mms-tts-tur) |

The project uses pretrained checkpoints from Meta's Massively Multilingual Speech (MMS) family, hosted on Hugging Face.

**Note:** This project generates speech in the selected language. It does not translate text between languages.

## Features

- Multilingual speech generation for four languages.
- Pretrained model integration using Hugging Face Transformers.
- Automatic selection of CPU or CUDA-enabled GPU when available.
- Model caching to avoid repeatedly loading the same checkpoint.
- WAV audio generation and export.
- Audio playback directly in Google Colab.
- Downloadable audio files.
- Basic audio diagnostics, including duration, peak amplitude, RMS amplitude, near-silence samples, and near-full-scale samples.
- Optional predicted speech-quality scoring using a pretrained CAL-MOS model.
- Manual listening ratings for acoustic fidelity, prosodic naturalness, intelligibility, and semantic clarity.

## Technologies Used

- **Python** — application logic and workflow.
- **PyTorch** — model inference and tensor operations.
- **Hugging Face Transformers** — loading tokenizers and pretrained TTS models.
- **NumPy** — waveform processing and numerical calculations.
- **SoundFile** — reading and writing audio files.
- **Google Colab** — notebook execution, audio playback, and file downloads.
- **CAL-MOS** — optional pretrained speech-quality prediction.

## Project Workflow

1. **Select a language:** Choose English, Hindi, Telugu, or Turkish.
2. **Enter text:** Provide the text that should be spoken in the selected language.
3. **Load the model:** Load the corresponding pretrained tokenizer and TTS checkpoint.
4. **Tokenize the input:** Convert text into tensors that the model can process.
5. **Generate speech:** Run inference to produce an audio waveform.
6. **Process the output:** Convert the waveform into a NumPy array and obtain the model's sample rate.
7. **Save and play audio:** Write the waveform to a WAV file and play it in Colab.
8. **Evaluate the audio:** Optionally inspect waveform statistics, predict an overall quality score, and record manual listening ratings.
9. **Download the output:** Download the generated WAV file for further use.

## Getting Started

### Prerequisites

- A Google account to access Google Colab.
- An internet connection to download the pretrained models.
- Basic familiarity with Python.
- A GPU is optional; the notebook can use a CPU when a CUDA-enabled GPU is unavailable.

### Run in Google Colab

1. Open the project notebook: `Multilingual_Text_to_Speech_HuggingFace.ipynb`.
2. Upload the notebook to Google Colab or open it from your GitHub repository.
3. Run the dependency-installation cell.
4. Run the import and device-configuration cell.
5. Execute the remaining cells in order.
6. Select a language and enter your text.
7. Run the speech-generation cell.
8. Listen to the generated audio and download the WAV file.

### Install Dependencies

Run this in a Google Colab code cell:

```python
%pip install --upgrade transformers torch soundfile
```

If the runtime was restarted, execute the notebook cells again from the beginning so that imports, variables, and model configuration are initialized.

### Example Usage

The notebook provides sample sentences for the supported languages. The following illustrates the main function call:

```python
language = "English"
text = "Hello! Welcome to my multilingual text-to-speech project."

output_path, waveform, sample_rate = generate_speech(
    text,
    language
)

print("Saved audio:", output_path)
print("Sample rate:", sample_rate, "Hz")
```

Play the generated audio in Colab:

```python
from IPython.display import Audio, display

display(Audio(waveform, rate=sample_rate))
```

Download the generated WAV file:

```python
from google.colab import files

files.download(output_path)
```

The notebook's `generate_speech()` function loads the selected language model, tokenizes the text, performs inference, and saves the generated waveform.

## Audio Quality Evaluation

Speech generation and speech-quality evaluation are separate tasks. The project includes three complementary approaches to examining generated audio.

### 1. Basic Audio Diagnostics

The `inspect_audio()` function reports:

- **Duration:** Length of the generated audio in seconds.
- **Peak amplitude:** Largest absolute audio-sample value.
- **RMS amplitude:** A measure of the waveform's overall signal energy.
- **Near-silence percentage:** Percentage of samples below a small amplitude threshold.
- **Near-full-scale percentage:** Percentage of samples close to the maximum normalized amplitude.

These measurements help identify potential signal-level issues. They do not independently measure speech naturalness, pronunciation accuracy, or intelligibility.

### 2. Optional CAL-MOS Prediction

The notebook optionally uses the pretrained checkpoint [`alefiury/CALMOS-WavLM-large-BVCC`](https://huggingface.co/alefiury/CALMOS-WavLM-large-BVCC) to predict an overall speech-quality score.

The predictor analyzes generated audio and estimates perceived quality. It does not generate speech or replace the TTS model.

Important considerations:

- The predicted score is not the same as a score collected from human listeners.
- Its reliability for Hindi, Telugu, and Turkish must be validated before drawing language-level conclusions.
- Predictions should be interpreted cautiously and should not be treated as definitive proof of speech quality.

### 3. Manual Listening Ratings

The notebook allows a listener to rate generated audio from 1 (poor) to 5 (excellent) on the following dimensions:

| Dimension | What it evaluates |
|---|---|
| Acoustic fidelity | Perceived clarity and audio quality |
| Prosodic naturalness | Rhythm and intonation |
| Phonetic intelligibility | Clarity of pronunciation |
| Semantic clarity | Whether the spoken output communicates the intended text clearly |

These ratings are exploratory. A more rigorous evaluation would use multiple sentences, multiple listeners, a consistent rating procedure, and saved results for analysis.

## Project Structure

```text
multilingual-text-to-speech/
├── Multilingual_Text_to_Speech_HuggingFace.ipynb
├── README.md
└── generated_audio/
    ├── speech_english.wav
    ├── speech_hindi.wav
    ├── speech_telugu.wav
    └── speech_turkish.wav
```

The `generated_audio/` directory is an example of how output files can be organized locally. The notebook generates WAV files when its cells are executed; the directory and example files do not necessarily exist in the repository by default.

## Key Learning Outcomes

Through this project, I explored:

- Using pretrained deep-learning models for inference.
- Loading language-specific checkpoints from Hugging Face.
- Text tokenization and PyTorch tensor processing.
- Running model inference with `torch.inference_mode()`.
- Managing CPU and GPU execution.
- Converting model outputs into audio waveforms.
- Saving and playing WAV files.
- Implementing basic audio-signal diagnostics.
- Understanding the difference between human listening ratings and model-predicted quality scores.
- Identifying evaluation limitations and responsible model usage.

## Limitations and Future Improvements

### Current Limitations

- The implementation relies on pretrained models rather than training a TTS model from scratch.
- Speech quality and pronunciation may vary by language and input text.
- The system does not provide automatic translation.
- The waveform diagnostics are basic signal measurements, not perceptual quality metrics.
- The optional quality predictor has not been established as reliable for every supported language.
- The notebook does not currently provide a complete deployed web application.
- Loading and retaining several language models can require substantial memory.

### Future Improvements

- Build an interactive interface using Streamlit.
- Add support for more languages where suitable pretrained models are available.
- Save diagnostic and listening-rating results to CSV for comparison.
- Evaluate multiple test sentences per language.
- Collect ratings from multiple human listeners.
- Measure model-loading time and speech-generation latency.
- Explore model optimization and deployment options.
- Validate the quality predictor against human ratings before making comparative claims.

## Model Licensing and Responsible Use

The four MMS-TTS model cards listed above identify their checkpoints under **CC-BY-NC 4.0**. This license generally requires attribution and restricts commercial use.

Before using these models in a paid application, commercial service, or business workflow, review the current model-card terms and obtain any additional permissions that may be required.

This repository is intended as an educational and portfolio project. Users are responsible for checking the applicable model licenses and the terms of any additional evaluation model they choose to use.

## Author

Developed as an AI/ML learning and portfolio project to explore pretrained speech-generation models, multilingual inference, and audio-quality evaluation.

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Meta MMS-TTS model collection](https://huggingface.co/models?search=facebook%2Fmms-tts)
- [CAL-MOS quality-prediction checkpoint](https://huggingface.co/alefiury/CALMOS-WavLM-large-BVCC)

---

**Disclaimer:** This project is an educational prototype and is not a guarantee of production-level speech quality. Generated audio and predicted quality scores should be evaluated before use in real-world applications.
