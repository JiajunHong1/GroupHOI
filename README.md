# GroupHOI
## Installation
Installl the dependencies.
```
pip install -r requirements.txt
```
Clone and build CLIP.
```
git clone https://github.com/openai/CLIP.git && cd CLIP && python setup.py develop && cd ..
```
## Data preparation

HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk) and the annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R).

V-COCO dataset can be downloaded [here](https://github.com/s-gupta/v-coco) and the prior file `prior.pickle` from [here](https://drive.google.com/drive/folders/10uuzvMUCVVv95-xAZg5KS94QXm7QXZW4).

For more details, check out the [GEN-VLKT](https://github.com/YueLiao/gen-vlkt).

## Pre-trained model
Download the pretrained model of DETR detector for [ResNet50](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth), and put it to the `params` directory.
```
python ./tools/convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-2branch-hico.pth \
        --num_queries 64

python ./tools/convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-2branch-vcoco.pth \
        --dataset vcoco \
        --num_queries 64
```

## Training
After the preparation, you can start training with the following commands.
### HICO-DET
```
sh ./configs/hico_s.sh
```

### V-COCO
```
sh ./configs/vcoco_s.sh
```

## Evaluation

### HICO-DET
You can conduct the evaluation with trained parameters for HICO-DET as follows.
```
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env \
        main.py \
        --pretrained exps/hico/checkpoint_best.pth\
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --eval \
        --with_clip_label \
        --with_obj_clip_label \
        --use_nms_filter
```
For the official evaluation (reported in paper), you need to covert the prediction file to a official prediction format following [this file](./tools/covert_annot_for_official_eval.py), and then follow [PPDM](https://github.com/YueLiao/PPDM) evaluation steps.
### V-COCO
Firstly, you need the add the following main function to the vsrl_eval.py in data/v-coco.
```
if __name__ == '__main__':
  import sys

  vsrl_annot_file = 'data/vcoco/vcoco_test.json'
  coco_file = 'data/instances_vcoco_all_2014.json'
  split_file = 'data/splits/vcoco_test.ids'

  vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)

  det_file = sys.argv[1]
  vcocoeval._do_eval(det_file, ovr_thresh=0.5)
```

Next, for the official evaluation of V-COCO, a pickle file of detection results have to be generated. You can generate the file with the following command. and then evaluate it as follows.
```
python generate_vcoco_official.py \
        --param_path pretrained/VCOCO_GroupHOI_S.pth \
        --save_path vcoco.pickle \
        --hoi_path data/v-coco \
        --num_queries 64 \
        --dec_layers 3 \
        --use_nms_filter \
        --with_clip_label \
        --with_obj_clip_label

cd data/v-coco
python vsrl_eval.py vcoco.pickle

```

## Regular HOI Detection Results

### HICO-DET
|                    | Full (D) |Rare (D)|Non-rare (D)|Full(KO)|Rare (KO)|Non-rare (KO)|Download| Conifg|
|:-------------------|:--------:| :---: | :---: | :---: |:-------:|:-----------:| :---: | :---: |
| GroupHOI-S (R50)   |  36.70   | 34.86 |37.26 | 39.42|  37.78  |    39.91    | [model](https://drive.google.com/file/d/1oIorpMv_7ohBjwnILqUHe8Clc-gV31-M/view?usp=drive_link) | [config](./configs/hico_s.sh)|

D: Default, KO: Known object


### V-COCO
| | Scenario 1 | Scenario 2 | Download | Config | 
| :--- | :---: | :---: | :---: | :---: |
|GroupHOI-S (R50)| 65.0| 66.0 | [model](https://drive.google.com/file/d/1uF-guUAj7wVx00WsP5FdRIEhKnljFnAu/view?usp=drive_link) |[config](./configs/vcoco_s.sh) |


## Citation
Please consider citing our paper if it helps your research.
```
@inproceedings{honglearning,
  title={Learning Human-Object Interaction as Groups},
  author={Hong, Jiajun and Wei, Jianan and Wang, Wenguan},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## License
GroupHOI is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

## Acknowledge
Some of the codes are built upon [PPDM](https://github.com/YueLiao/PPDM), [DETR](https://github.com/facebookresearch/detr), [QPIC](https://github.com/hitachi-rd-cv/qpic) and [CDN](https://github.com/YueLiao/CDN),[GEN-VLKT](https://github.com/YueLiao/gen-vlkt). Thanks them for their great works!
