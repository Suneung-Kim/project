# Project_1

## Requirements
python >= 3.7 or 3.8 my version is 3.7.10 우분투 환경 20.04

```bash
pip install -r requirement.txt
```
## Training

```bash
python train.py --config_file configs/se_resnet18.yaml
```
## Test

```bash
python test.py --config_file configs/se_resnet18.yaml
```
## Cam_test
```bash
python cam_test.py --model se_resnet18
```

pretrained model은 밑의 링크에서 다운받을 수 있습니다.

https://drive.google.com/file/d/1FV0SMQLFP-t17sJ8UuoCb03EAOU_ga0w/view?usp=sharing

dataset

https://drive.google.com/file/d/1Lxlwiz2xXBBWJCTX4S1TbAS-JMRoni7Z/view?usp=sharingdataset

## Se_resnet training result

![학습결과](https://user-images.githubusercontent.com/70845599/141709257-be45b64a-9395-4850-8104-9a3f65db58d1.png)

## Test result

![test 결과](https://user-images.githubusercontent.com/70845599/141709382-f940e7c6-7129-4e27-bdb0-4768e2100ad5.png)

## Cam_result


