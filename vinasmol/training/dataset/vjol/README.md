# CCVJ - CreativeCommons Vietnamese Journals Dataset

We compile a new dataset by downloading 12000 academic papers from 18 permissively-licensed journals referenced by [DOAJ](https://doaj.org) and/or [VJOL](https://vjol.info.vn) and processing them with [Docling](https://github.com/docling-project/docling).

## To reproduce

### Determine the harvesting sources

To download the [OAI-PMH](https://www.openarchives.org/pmh/) API endpoints for harvesting, run the Jupyter notebook `doaj_vn.ipynb`.

This notebook should create the file `./data/vjol/records/api_urls.json`.

### Download the papers

The [main script](./__main__.py) will download the papers, process the PDFs and save the newly created dataset to be used for SmolLM Vietnamese training data. You can launch the script with the following Python command:

```bash
python -m vjol --simultaneous-downloads 8
```

Downloading metadata and the papers can take a few hours. Expect around 50GB of PDF downloads for approximately 100M Vietnamese tokens. Conversion to Markdown should take a few days on a single-GPU machine.

The script will resume where it left off if you execute it a second time.

### Compress the PDFs

You need the [GhostScript](https://www.ghostscript.com) command (`gs`) available on your `PATH` environment variable.

```bash
python compress_pdfs.py data/vjol/pdf
```

## Licenses

We did our best to find the individual licenses for every paper in the dataset. All of the works in CCVJ satisfy the following conditions:
1. The publishing journal operates under a compatible license.
2. The paper page specifies a compatible license, if provided.

The compatible licenses are:

- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0)
- [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0)
- [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0)
- [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)