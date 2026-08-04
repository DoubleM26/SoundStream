# SoundStream Neural Audio Codec

A PyTorch research implementation of a neural audio codec for speech, based on
[SoundStream](https://arxiv.org/abs/2107.03312) and its
[SEANet](https://arxiv.org/abs/2009.02095) training setup. The model compresses
16 kHz audio into a discrete representation at a nominal bitrate of 6.4 kbps
and reconstructs the waveform with a neural decoder.

[Colab demo](https://colab.research.google.com/drive/1p-2ddMwsf0EMib1FKh6CH70RdlemgmNT?usp=sharing)
| [Pretrained model](https://huggingface.co/mishgun100/soundstream)
| [Experiment report (Russian)](https://www.comet.com/doublem26/dl-soundstream/reports/zNdxQmn4FpypxjJbW4lwJ2hwZ)

## Highlights

- Implemented the causal encoder and decoder, residual vector quantization,
  waveform and STFT discriminators, and the complete training pipeline.
- Built the residual vector quantizer from scratch, including EMA codebook
  updates, dead-code replacement, and straight-through estimation.
- Trained for 45,000 steps on the 100-hour LibriSpeech `train-clean-100` split.
- Logged losses, codebook statistics, reconstructed audio, and validation
  metrics with Comet ML.
- Evaluated the final model on all 2,620 utterances from LibriSpeech
  `test-clean`.

## Results

| Dataset | STOI | NISQA MOS |
| --- | ---: | ---: |
| LibriSpeech `test-clean` | **0.8086** | **2.3461** |

The [Colab demo](https://colab.research.google.com/drive/1p-2ddMwsf0EMib1FKh6CH70RdlemgmNT?usp=sharing)
downloads the pretrained checkpoint, reconstructs an external audio sample,
and plays the original and reconstructed versions. Additional training curves,
audio samples, and analysis are available in the
[experiment report](https://www.comet.com/doublem26/dl-soundstream/reports/zNdxQmn4FpypxjJbW4lwJ2hwZ).

## Usage

```bash
git clone https://github.com/DoubleM26/SoundStream.git
cd SoundStream
pip install -r requirements.txt
python train.py configs/train.yml
```

The final checkpoint is available on
[Hugging Face](https://huggingface.co/mishgun100/soundstream).

## Scope

This project was originally developed as part of a Deep Learning course.

