# Multilingual contradiction detection

**Authors:** Jérémie Dentan (jeremie.dentan@live.com) and Alicia Rakotonirainy (alicia.rakotonirainy@gmail.com)

## Overview

This repository implements the training of models to perform multilingual contradiction detection. Please refer to [our report here](doc/report.pdf) for more details about the method we used.

## Run the code

The data and the model will be automatically downloaded from the internet when needed. This code is expected to be run under `Python 3.8` with the dependencies stated on `requirements.txt` installed and the `PYTHONPATH` set at the root of the project. To do so, run the following from the root of the project.

```bash
export PYTHONPATH=$(pwd)
pip install -r requirements.txt
```

The running generates logs in `/logs/full.log` and `/logs/summary.log` that you can check to monitor your runs.

### Run the training

The training took about 2 hours on NVIDIA GeForce RTX 3090 and requires at least 6 Go of graphic memory. To run it, execute the following line from the root. This will generate features in `/output/features`.

```bash
python -m src.features
```

### Run the prediction

To run the prediction with XBGoost based on the features you computed, run the following line from the root. This will generate a submission in `/output/submissions`.

```bash
pytyhon -m src.predict
```
