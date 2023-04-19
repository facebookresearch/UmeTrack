# UmeTrack: Unified multi-view end-to-end hand tracking for VR

## Introduction
This is the project page for the paper [UmeTrack: Unified multi-view end-to-end hand tracking for VR](https://research.facebook.com/publications/umetrack-unified-multi-view-end-to-end-hand-tracking-for-vr/). The *pre-trained inference model*, **sample code for running the inference model** and **the dataset** are all included in this project.

## Environment setup
```bash
conda create --name umetrack python=3.9.12
conda activate umetrack
pip install av numpy scipy opencv-python "git+https://github.com/facebookresearch/pytorch3d.git@stable" 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## UmeTrack_data
The data is stored in the submodule *UmeTrack_data*.
* `raw_data` contains the raw images from the 4 fisheye sensors placed on a headset. Each recording consists of a mp4 file and a json file.
* `torch_data` is used when we train the models. The data are generated from `raw_data` but are packed to be more friendly for batching during training.
  * `mono.torch.bin` contains the image data that have been resampled using pinhole cameras.
  * `labels.torch.bin` contains the hand pose labels
  * `mono.torch.idx` and `labels.torch.idx` are indices into the above 2 files to allow random access to the data without reading data into memory.

## Running the code
Run evaluations using **known** skeletons on `raw_data`
```
python run_eval_known_skeleton.py
```

Run evaluations using **unknown** skeletons on `raw_data`
```
python run_eval_unknown_skeleton.py
```

Gather evaluation results for `raw_data`
```
python load_eval.py
```

Run evaluations using on `torch_data`
```
python run_inference_torch_data.py
```



## Results
Ours results are compared to [[Han et al. 2020]](https://research.facebook.com/publications/megatrack-monochrome-egocentric-articulated-hand-tracking-for-virtual-reality/). There are some minor differentces between the metrics here and Table 3 in the main paper.
* The formula we used internally was slightly different from **MPJPA** metric and we made a mistake in putting those numbers in the main paper. The table below is updated using **eq. 10** introduced the main paper.
* The skeleton calibration has been improved compared to when we published the paper. As a result, we are showing superior results in the **Unknown hand skeleton** category.

<table class="center">
<thead>
  <tr>
    <th class="tg-0lax">Method</th>
    <th class="tg-0lax" colspan="4"> Known hand skeleton</th>
    <th class="tg-0lax" colspan="4">Unknown hand skeleton</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax"></td>
    <th class="tg-0lax" colspan="2">separate-hand</th>
    <th class="tg-0lax" colspan="2">hand-hand</th>
    <th class="tg-0lax" colspan="2">separate-hand</th>
    <th class="tg-0lax" colspan="2">hand-hand</th>
  </tr>
  <tr>
    <td class="tg-0lax"></td>
    <td class="tg-0lax">MPJPE</td>
    <td class="tg-0lax">MPJPA</td>
    <td class="tg-0lax">MPJPE</td>
    <td class="tg-0lax">MPJPA</td>
    <td class="tg-0lax">MPJPE</td>
    <td class="tg-0lax">MPJPA</td>
    <td class="tg-0lax">MPJPE</td>
    <td class="tg-0lax">MPJPA</td>
  </tr>
  <tr>
    <td class="tg-0lax">[Han et al. 2020]</td>
    <td class="tg-0lax">9.9</td>
    <td class="tg-0lax">4.63</td>
    <td class="tg-0lax">10.8</td>
    <td class="tg-0lax">4.09</td>
    <td class="tg-0lax">12.9</td>
    <td class="tg-0lax">4.67</td>
    <td class="tg-0lax">13.6</td>
    <td class="tg-0lax">4.17</td>
  </tr>
  <tr>
    <td class="tg-0lax">Ours</td>
    <td class="tg-0lax">9.4</td>
    <td class="tg-0lax">3.92</td>
    <td class="tg-0lax">10.6</td>
    <td class="tg-0lax">3.47</td>
    <td class="tg-0lax">10.0</td>
    <td class="tg-0lax">3.86</td>
    <td class="tg-0lax">10.9</td>
    <td class="tg-0lax">3.44</td>
  </tr>
</tbody>
</table>

## Reference
```
@inproceedings{han2022umetrack,
  title = {UmeTrack: Unified multi-view end-to-end hand tracking for {VR}},
  author = {Shangchen Han and Po{-}Chen Wu and Yubo Zhang and Beibei Liu and Linguang Zhang and Zheng Wang and Weiguang Si and Peizhao Zhang and Yujun Cai and Tomas Hodan and Randi Cabezas and Luan Tran and Muzaffer Akbay and Tsz{-}Ho Yu and Cem Keskin and Robert Wang},
  booktitle = {{SIGGRAPH} Asia 2022 Conference Papers, {SA} 2022, Daegu, Republic of Korea, December 6-9, 2022},
  year = {2022}
}
```

## License
UmeTrack is licensed under the Creative Commons Attribution-NonCommerial 4.0 International License, as found in the LICENSE file.

