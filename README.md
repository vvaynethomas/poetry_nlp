# poetry_project

## Analytics and Text Generation of Poetry Foundation Dataset

These are snippets and notebooks of things that the total platform may comprise, exploring and modeling a Poetry Foundation corpus. At the drafting stage, I allowed a meandering to gain insights and see what was feasible in the timeframe. I explored the data visually, experimented with embeddings and different methods for analyzing and generating text, and ran into memory limitations while attempting to create padded line arrays for TensorFlow and NLTK tokenizers. The longer-term possibilities in the original project framing included semantic and structural analysis and context-specific text generators; those possibilities are not claims that a finished system was built.

## Start here

Begin with [`midterm_notebook.ipynb`](midterm_notebook.ipynb), the broadest record of the data preparation, feature exploration, part-of-speech tagging, Doc2Vec, and clustering workflow. Its saved cell outputs are historical notebook state, not a fresh run.

| Notebook or file | What it contains |
| --- | --- |
| [`preprocessing.ipynb`](preprocessing.ipynb) | Earlier dataset cleaning and preparation experiments. |
| [`preproc_and_ner.ipynb`](preproc_and_ner.ipynb) | Additional preprocessing and text-analysis experiments. |
| [`poems2doc2vec.ipynb`](poems2doc2vec.ipynb) | Doc2Vec preparation/training using tagged poem and line data. |
| [`poem_clustering.ipynb`](poem_clustering.ipynb) | Poem-vector work and clustering experiments. |
| [`syntactic_analysis.ipynb`](syntactic_analysis.ipynb) | Analysis of saved line-level tags. |
| [`nltk_keras.ipynb`](nltk_keras.ipynb), [`structural_modeling.ipynb`](structural_modeling.ipynb), [`nlg.ipynb`](nlg.ipynb), [`line_predicter.ipynb`](line_predicter.ipynb) | Sequence-model and text-generation/prediction experiments. |
| [`nlg_mp.py`](nlg_mp.py) | Small Keras/NLTK multiprocessing-related helper. |
| [`Midterm_report_presentation.pdf`](Midterm_report_presentation.pdf), [`w_yandell_midterm_report.pdf`](w_yandell_midterm_report.pdf), [`midterm_writeup.docx`](midterm_writeup.docx) | Historical project reports and presentation. |

The notebooks are exploratory and overlap; they are not a single, ordered production pipeline. The reports describe goals such as coherent poem generation, while the repository records experiments toward those goals rather than a verified end-to-end generator.

## Data provenance and reuse

The report identifies the source as the Poetry Foundation website dataset distributed by John Hallman on Kaggle: [Complete Poetry Foundation Dataset](https://www.kaggle.com/johnhallman/complete-poetryfoundationorg-dataset). The notebooks reference a local `kaggle_poem_dataset.csv`. That raw CSV and `archive.zip` have been removed from this repository pending rights review. The historical ZIP contained a CSV that was not byte-for-byte identical to the root CSV; the source/version relationship was not documented.

No dataset license or permission record is included here. The presence of derived files in this repository is not a determination that they may be redistributed; serialized data may contain corpus text. Their source, reuse rights, and continued inclusion need maintainer review. Do not assume that public availability of the source or repository grants those rights.

## Saved results and reproduction status

Notebook outputs and saved artifacts represent historical work. The repository contains pickled corpora/dataframes, Doc2Vec models and vectors, and an HDF5 sequence-model file. They are not newly reproduced or independently validated results, and no new metrics are claimed here.

A clean reproduction was not performed. There is no dependency manifest or documented setup, the raw CSV expected by the notebooks is not included, and the tracked `venv/` referenced a machine-specific Python 3.8 environment that is no longer present. Notebook execution may also require language-model or tokenizer resources and substantial memory/GPU capacity. Until permitted data access and a supported setup are documented and run, the saved outputs should not be treated as proof that the notebooks execute end to end today.

## Repository contents and housekeeping

- The notebooks reference `kaggle_poem_dataset.csv`, which is not included. Use the cited Kaggle source subject to its terms and any applicable rights review. The non-identical `archive.zip` copy was also removed.
- Pickles, `.npy` vectors, Doc2Vec models, and the `.hdf5` model file remain as historical research artifacts pending the same rights review; they are not disposable build output.
- `.ipynb_checkpoints/` contains tracked notebook snapshots and `.idea/` contains tracked IDE metadata. Both are preserved; ignore rules prevent newly generated local copies from being added.
- The tracked `venv/` was a local environment tied to a missing absolute Python path, not a portable setup. It has been removed; no replacement environment or dependency set is claimed.

No license for this project or dataset is asserted by this README.
