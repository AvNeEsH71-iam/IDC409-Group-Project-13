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
    |-- text_tasks.py      (person 1)
    |-- features.py        (person 2)
    |-- train_model.py     (person 2)
    |-- evaluate.py        (person 3)
    |-- compare_conditions.py(person 3)
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

We use **faster-whisper** (a fast CPU implementation of OpenAI Whisper) with two pretrained English models,
`tiny.en` and `base.en`, so that two ASR systems can be compared. Nothing is trained.

For each clip `src/transcribe.py`:

1. decodes the audio and runs the model (`beam_size=5`, language `en`);
2. normalises the text (lowercase, no punctuation) so it can be compared with the LibriSpeech transcript;
3. computes a **confidence** score = duration-weighted `exp(avg_logprob)` over the segments (0 to 1);
4. writes **notes** (`empty transcript`, `low confidence` when confidence < 0.60);
5. saves the result to the database in batches, so an interrupted run can simply be started again (finished clips are skipped).

```bash
python -m src.transcribe --model tiny.en
python -m src.transcribe --model base.en
python -m src.transcribe --model tiny.en --limit 5     # quick test
```

Outputs: `outputs/transcripts_tiny.en.csv` / `.txt` and `outputs/transcripts_base.en.csv` / `.txt`
(CSV columns: clip_id, filename, condition, snr_db, true_text, predicted_text, confidence, notes).
On a normal laptop CPU, expect roughly 10 to 30 minutes per model for the default 240 clips.

## 3. WER evaluation and EDA

### Word error rate

`WER = (substitutions + deletions + insertions) / words in the true transcript`, computed per clip with
`jiwer` after normalising both texts. We report the mean per-clip WER and the corpus WER (total errors / total words).

```bash
python -m src.metrics        # writes WER to the database, outputs/wer_per_clip.csv, outputs/wer_summary.csv
```

### Exploratory data analysis

```bash
python -m src.eda
```

Creates `outputs/eda_summary.txt` (counts, describe() tables, missing values, top error examples),
`outputs/error_examples.csv` (15 worst clips with true vs predicted text) and the plots in `outputs/plots/`:
clip duration, word count, transcript length vs duration, WER by condition and ASR model, and confidence vs WER.

### Limitations (so far)

* LibriSpeech is read book speech: clean, mostly American / British, so it is easier than real conversations.
* The subset is small (60 clips x 4 conditions), so differences between models can be noisy.
* LibriSpeech writes numbers as words while Whisper may output digits, which counts as an error in WER.
* Noise is synthetic white noise; real background noise (traffic, crowds) behaves differently.
* Everything runs on CPU with small Whisper models; larger models would be more accurate but slower.

## 4. Text analysis (keyword detection and answer matching)

```bash
python -m src.text_tasks
```

**Keyword detection.** The 20 most common content words (document frequency, stop words and words shorter than 4 letters removed)
of the true transcripts become the keyword list (`outputs/keywords.txt`). For each clip, a keyword counts as present if it
appears in the transcript. The *true* transcript is the ground truth and the *predicted* transcript is the detection, so we get
true positives, false positives and false negatives, and from them **precision, recall and F1** per ASR model and condition
(`outputs/keyword_metrics.csv`) and per keyword (`outputs/keyword_per_word.csv`).

**Matching to the expected answer.** Each predicted transcript is compared with the expected (true) transcript using
`difflib.SequenceMatcher` similarity. A clip is an *exact match* when the normalised texts are identical and a *fuzzy match* when
similarity >= 0.85. We report exact accuracy, fuzzy accuracy and mean similarity
(`outputs/matching_results.csv`, `outputs/matching_summary.csv`).

## 5. Features, feature reduction and machine-learning models

to be done by person 2

_Planned content: features, feature reduction and machine-learning models._

## 6. Evaluation, extension and results

to be done by person 3

_Planned content: final evaluation, extension, results, limitations and team contributions._
