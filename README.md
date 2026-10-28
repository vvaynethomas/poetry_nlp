# Poetry NLP: exploring poetic form and language

How can a poem be represented as data without losing sight of its lines, vocabulary, and structure? This exploratory Python project studies a Poetry Foundation corpus through descriptive analysis, part-of-speech tagging, document embeddings, and clustering, then experiments with sequence models for text and line prediction.

The interesting thread is the shift between scales: whole-poem representations for similarity and clustering, and line-level features for examining structure. This is a notebook-based research archive, not a finished poetry generator or a packaged application. The original work followed promising questions and practical constraints rather than a single production pipeline.

## Start exploring

**Start with [`midterm_notebook.ipynb`](midterm_notebook.ipynb)** for the broadest walkthrough: cleaning, visual exploration, tagging, Doc2Vec, and several clustering approaches. You can read its saved outputs on GitHub without installing Python or loading any model files.

For a shorter overview, open the [presentation](Midterm_report_presentation.pdf) or [written report](w_yandell_midterm_report.pdf). The [DOCX writeup](midterm_writeup.docx) is also preserved. These are historical accounts of the project, including its aspirations; they are not evidence of a finished end-to-end generator.

## What the notebooks explore

| Question | Approach |
| --- | --- |
| What does the corpus look like at poem and line level? | pandas-based deduplication and cleaning; distributions of line counts, word counts, word lengths, and lexical diversity; matplotlib and seaborn visualizations. |
| How can grammatical structure be represented? | spaCy part-of-speech tagging and separate experiments with saved line-level tags. |
| Can poems be compared beyond surface counts? | Gensim Doc2Vec using distributed bag-of-words (`dm=0`), with 500- and 2,000-dimensional training experiments. |
| What groupings appear in the embeddings? | Scaling, normalization, PCA, and comparisons of K-means, DBSCAN, agglomerative clustering, and Gaussian mixture models; elbow and silhouette diagnostics. |
| Can sequence models extend the analysis toward generation? | TensorFlow/Keras token-sequence experiments and a bidirectional LSTM for predicting line vectors. |

### A few observations from the saved work

These are **historical notebook outputs**, not freshly reproduced findings or model-performance claims:

- In the midterm notebook's **Parse, Explore, and Filter** section, the source table has 15,652 rows. Removing duplicate content, duplicate author/title pairs, and missing values leaves 15,607 rows.
- The line-level analysis makes the effect of filtering visible: its dataframe summaries show 433,998 lines before the word-count outlier filter and 431,229 afterward. These are intermediate dataset sizes, not counts for every later experiment.
- In the **KMeans** section, the elbow detector returns nine clusters for that particular representation and run. This is an exploratory diagnostic, not validation of nine poetic genres or a best-performing model.
- The generation experiments expose a practical bottleneck: `nlg.ipynb` records a memory-allocation failure while creating one-hot targets, before a successful end-to-end generation workflow is established.

## Notebook map

The order below is a reading guide, not an execution dependency order. The notebooks overlap and sometimes consume previously saved artifacts.

| File | Focus |
| --- | --- |
| [`midterm_notebook.ipynb`](midterm_notebook.ipynb) | Main walkthrough, from raw CSV preparation to embedding and clustering comparisons. |
| [`preprocessing.ipynb`](preprocessing.ipynb) | Earlier dataset cleaning and preparation experiments. |
| [`preproc_and_ner.ipynb`](preproc_and_ner.ipynb) | Additional preprocessing experiments and text-analysis setup. |
| [`poems2doc2vec.ipynb`](poems2doc2vec.ipynb) | Tagged-document preparation and 500-dimensional Doc2Vec training. |
| [`poem_clustering.ipynb`](poem_clustering.ipynb) | Vector reduction, clustering comparisons, and diagnostic plots. |
| [`syntactic_analysis.ipynb`](syntactic_analysis.ipynb) | Exploration of saved line-level part-of-speech tags. |
| [`nltk_keras.ipynb`](nltk_keras.ipynb), [`structural_modeling.ipynb`](structural_modeling.ipynb) | Tokenization and structural/sequence-model preparation experiments. |
| [`nlg.ipynb`](nlg.ipynb) | Token-sequence preparation and an LSTM text-generation model definition, with saved errors. |
| [`line_predicter.ipynb`](line_predicter.ipynb) | Doc2Vec line encoding and bidirectional LSTM line-vector prediction, including training and model saving. |
| [`nlg_mp.py`](nlg_mp.py) | A small import-only fragment, not a runnable application or pipeline. |

## Local exploration and setup status

**Reading is the supported starting point; a clean execution environment has not been verified.** There is no dependency manifest or pinned package set. For local inspection, use a notebook viewer or an existing Jupyter installation. From the repository directory, this opens the main notebook; it does not run its cells:

```shell
jupyter notebook midterm_notebook.ipynb
```

Before attempting execution, inspect the notebook's imports and file-loading cells. The main analysis uses pandas, NumPy, SciPy, NLTK, spaCy, Gensim, scikit-learn, matplotlib, seaborn, tqdm, and kneed; the sequence experiments additionally use TensorFlow/Keras. This is an orientation list, not a tested installation recipe. Language resources include spaCy's `en_core_web_sm` and NLTK tokenizer/corpus resources. Older APIs may need compatible library versions.

Notebooks expect local inputs such as `kaggle_poem_dataset.csv` and saved intermediate files in the working directory. Opening a notebook is not the same as reproducing its outputs: do not use **Run All** until you have reviewed its inputs, environment, and resource requirements.

## Data and saved artifacts

The documented source is John Hallman's [Complete Poetry Foundation Dataset on Kaggle](https://www.kaggle.com/johnhallman/complete-poetryfoundationorg-dataset), drawn from the Poetry Foundation website. The raw `kaggle_poem_dataset.csv` and `archive.zip` are not included. Consult the source's terms and applicable permissions before obtaining or reusing the data; this repository does not establish redistribution rights or assert a project or dataset license.

Pickled corpora/dataframes, Doc2Vec models and `.npy` vectors, and `line_sequence_lstm.final.hdf5` remain as historical research artifacts. Derived data may contain corpus text, so its presence is not a grant of reuse rights. **Do not unpickle files or load serialized models from an untrusted source:** loading them can execute code or require unsafe legacy deserialization. None is needed just to browse the notebooks.

## Limits to keep in mind

The corpus and cleaning choices bound the conclusions: deduplication, text normalization, and length-based filtering affect which poetic forms remain visible. Clusters are exploratory groupings, not established literary categories, and the archive does not demonstrate coherent poem generation or measured improvements over a baseline.

Saved outputs can reflect earlier notebook state. Some notebooks contain errors or interrupted cells, and the generation work encountered substantial memory demands. The missing raw CSV, unpinned dependencies, and legacy APIs prevent a claim that the notebooks run end to end today. Treat the archive as a record of methods, experiments, and tradeoffs rather than a turnkey reproducible system.
