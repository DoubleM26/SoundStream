# SoundStream Neural Audio Codec

This repo is an implementation of a SoundStream neural audio codec for speech.

[Demo notebook](https://colab.research.google.com/drive/1p-2ddMwsf0EMib1FKh6CH70RdlemgmNT?usp=sharing)

The demo notebook loads the model from HuggingFace, runs reconstruction on an external audio example, and plays both original and reconstructed audio.


## Installation 

```
git clone https://github.com/DoubleM26/SoundStream.git
cd SoundStream
pip install -r requirements.txt
```

## Training & Checkpoint

The final model was trained with `configs/train.yml` config  on LibriSpeech `train-clean-100` dataset. Full evalution on `test-clean` achieve: 
- STOI: 0.8086
- NISQA MOS: 2.3461

The final checkpoint is hosted on [HuggingFace](https://huggingface.co/mishgun100/soundstream).



## Evaluation

Evaluation runs on full audio from LibriSpeech `test-clean`:

```
python eval.py configs/train.yml
```

The checkpoint path is read from `eval.checkpoint_path` inside the config.

