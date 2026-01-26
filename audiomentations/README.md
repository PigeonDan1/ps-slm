# Audiomentations

Adopt from [audiomentations](https://github.com/iver56/audiomentations), thanks very much!

Currently, supported transformation contains four categories, pre_process, add_noise, impose_response, post_process.
Each category include serveral transforms.

For detail:
* pre_process
  * resample
  * volume_perturb
  * speed_perturb
  * pitch_perturb
  * concat
  * dump
  * change_encode (TODO)
  * eq_perturb
* add_noise
  * add_background_noise
  * pad_silence
* impose_response
  * apply_impose_response
* post_precess
  * denoise
    * cstub_aec
    * cstub_gsc
    * omlsa_nr
    * nnmask
  * dereverb
    * wpe
  * select_beam
  * select_channel
  * time_trim

There is one special transform, **MutuallyExclusiveGroup**, which contains serveral transforms plus one EmptyTransform. When this transform is performed, one of included transform is selected by configure probablity.

## Requirements
use conda environment: /mnt/lustre02/jiangsu/aispeech/home/hl219/tools/miniconda3/envs/pytorch

```
. ./path.sh
```

or install requirements personnally

```
python -m pip install -r requirements.txt --user
```

if you use nnmask in post process
```
git clone https://git.aispeech.com.cn/pytorch-asr/neural-beamforming.git NeuralSignal
ln -s NeuralSignal/nsp extend_codes
```

## Augmentation

Transformations could be squentially composed as a chain, which could be used to augment wavs.

The chain could be defined by yaml config file, like `conf/base.yaml`.

```yaml
input_wavlist: example/data/clean/wav.scp
output_dir: example/data/augment_base
transforms:
  - name: pitch_perturb
  - name: speed_perturb
```


Run in local in 1 process:

```bash
. ./path
python -u -m audiomentations.launch -c conf/base.yaml
```

To run the augmentation distributedly,

```bash
./scripts/run.sh --nj 4 -c conf/base.yaml 
```

## Cases
### taihang mono phone simu

For instance, taihang mono phone data augmentation process. The all process contains 4 main blocks, each block include several transforms.

![taihang_mono_phone_simu](assets/taihang_mono_phone_simu.png)

You could run this augmentation process by,

```bash
. ./path
python -m audiomentations.launch -c conf/taihang_mono_phone_simu.demo.yaml
```
