# Flow2Surf

Jittor implementation of **Point Cloud Denoising via Continuous Velocity Learning**.

The [competition version](https://github.com/Gearoid522/jittor-Flow2Surf)
ranked first on both A and B leaderboards of the Sixth Jittor Artificial Intelligence Challenge.

## Setup

Requires Linux, an NVIDIA GPU with CUDA support, and `g++`.

```bash
conda env create -f environment.yaml
conda activate flow2surf
```

Download [data.zip](https://cloud.tsinghua.edu.cn/f/bb3ba07d11ca45f78654/?dl=1)
and extract it at the repository root. Save
[final.pkl](https://cloud.tsinghua.edu.cn/f/33596ea4a36c40dc832a/?dl=1)
under `checkpoints/`.

## Training

Settings: [configs/default.yaml](configs/default.yaml).

```bash
python train.py
python train.py --resume path/to/epoch_010.pkl
```

Outputs are saved under `outputs/<dataset>_<timestamp>/`.
Use `epoch_*.pkl` to resume training and `final.pkl` for prediction and evaluation.

## Prediction

```bash
python predict.py --model checkpoints/final.pkl \
  --input cloud.xyz --passes 1 --out denoised.xyz
```

## Evaluation

```bash
python eval.py --dataset datasets/PUNet.yaml \
  --model checkpoints/final.pkl \
  --noise 1 --passes 1 --out results/pu20_1.0.json

python eval.py --dataset datasets/PUNet.yaml \
  --model checkpoints/final.pkl \
  --noise 2 --passes 2 --out results/pu20_2.0.json

python eval.py --dataset datasets/PUNet.yaml \
  --model checkpoints/final.pkl \
  --noise 2.5 --passes 3 --out results/pu20_2.5.json
```

Use `datasets/PCNet.yaml` for PC10. Results contain CD, P2M and prediction time.

## License

The code is released under the [MIT License](LICENSE).
