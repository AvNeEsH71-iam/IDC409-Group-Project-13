# Speech to Text: Transcribe English Audio, Then Analyse the Text

Group project (3 members). We convert English speech to text with a **pretrained** ASR model
(Whisper through `faster-whisper`, no training from scratch), store the data with numpy / pandas and a
SQLite database, run exploratory data analysis, and then work on the transcripts: keyword detection,
matching to the expected answer, and machine-learning models that predict where the ASR system struggles.

## 1. Project overview, data and setup

### Pipeline

```
data_prep -> transcribe -> metrics (WER) -> eda -> text_tasks -> features -> train_model -> compare_conditions -> evaluate
```

### Folder structure

```
speech-to-text-project/
|-- README.md
|-- requirements.txt
|-- data/            audio subset, manifest.csv, SQLite database, sample_audio/
|-- outputs/         transcripts, metrics, plots, models
`-- src/
    |-- config.py          (person 1) paths and settings
    |-- text_utils.py      (person 1) text normalisation helpers
    |-- database.py        (person 1) SQLite read / write / modify
    |-- data_prep.py       (person 1)
    |-- transcribe.py      (person 2)
    |-- metrics.py         (person 3)
    |-- eda.py             (person 3)
```

### Dataset

* **LibriSpeech dev-clean** (https://www.openslr.org/12): `src/data_prep.py` streams the archive and keeps a small
  subset (default 60 clips, at most 10 per speaker, each at most 15 s) as 16 kHz mono wav files.
  The archive is read as a stream and the download stops as soon as enough clips are found.
* **Noisy copies**: every clean clip is copied with white noise at 10, 5 and 0 dB SNR (numpy), giving four
  conditions: `clean`, `noisy_10dB`, `noisy_5dB`, `noisy_0dB`.
* **Optional own recordings** (e.g. Indian-accented speech): record wav/flac clips, put them in a folder with a
  CSV (`filename,true_text`) and run `python -m src.data_prep --skip-download --own-csv your_folder/recordings.csv`.
  They get the condition `accent_indian` and flow through every later step automatically.
* `data/sample_audio/` holds 5 clean clips that are committed to GitHub; the rest of the audio is re-created by the script.

For every clip we store: audio filename, true transcript, predicted transcript, confidence and notes
(table `transcripts` in the database and `outputs/transcripts_<model>.csv`).

### Setup

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9+ is needed. `faster-whisper` downloads the Whisper model files the first time it runs
(internet needed once; about 75 MB for `tiny.en`, 145 MB for `base.en`). No separate ffmpeg install is needed.

### Database (read, write, modify)

`src/database.py` uses SQLite (`data/speech_project.db`) with two tables:

| table | columns |
|---|---|
| `clips` | clip_id, source_clip_id, filename, filepath, condition, snr_db, duration_s, sample_rate, true_text |
| `transcripts` | id, clip_id, asr_model, predicted_text, confidence, notes, wer |

Functions: `insert_clips`, `insert_transcripts` (write), `read_clips`, `read_transcripts`, `read_joined` (read),
`update_notes`, `update_wer` (modify), `delete_transcripts`, `delete_clip` (delete).
`python -m src.database` runs a short read -> modify -> restore demo.

### Run the data steps

```bash
python -m src.data_prep            # download subset, create noisy versions, fill manifest + database
python -m src.database             # optional: check the database
```

## 2. Speech-to-text (ASR)

to be done by person 2

_Planned content: how the speech-to-text step works and how to run it._

## 3. WER evaluation and EDA

to be done by person 3

_Planned content: WER evaluation, EDA and current limitations._
