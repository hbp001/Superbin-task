# Multi Label Image-classification---Superbin
## Description
#### 재활용 쓰레기 종류를 분류하는 Task

## DB 및 backbone
#### DB : Superbin dataset
#### backbone code - ResNet, VGG, Resnet 


## Result
#### 최고 성능
1. ResNet18
2. batch_size = 32
3. learning_rate = 0.0125
4. epoch = 30
5. Augmentation : Resize((224,224)), RandomHorizontalFlip()
6. normalize (0.5)
7. loss = 0.0041, val_acc = 99.12
### Model Select

 Class의 개수가 4개로 많지 않고, 학습 시간도 굉장히 오래 걸리는 Task였기 때문에 비교적 작은 크기의 model인 resnet18을 선택했습니다.

### Loss Module

 이번 task가 multi label classification이었기 때문에 loss를 어떻게 넣어주느냐가 핵심이 되었습니다. Loss는 BCEWithLogitsLoss 와 MultiLabelSoftMarginLoss 두 가지 module이 있었는데, MultiLabelSoftMarginLoss를 선택했습니다. Multi Label Classification 문제에서는 label 별로 계산된 loss값이 개별적으로 쓰일 필요가 없기 때문에 BCE 값을 평균 내는 module인 MultiLbelSoftMarginLoss를 사용했습니다

### Hyper Parameter

[제목 없음](https://www.notion.so/6fead37708954e298db0275a4930543a)

- 1epoch을 진행이 너무 오래 걸려 30epoch만 진행했습니다.
- Batch size는 32를 넘으면 CUDA out of memory error가 발생하기 때문에 32로 고정해 주었습니다.
- Learning rate는 1회차에는 임의로 적당한 값을 주었고, 2, 3회차는 bag of trick을 활용해 1x(b/256)을 이용해 계산해 설정했습니다.
- 3회차에서는 epoch을 30만큼 실행하기 때문에 learning rate drop epoch을 (10, 20, 30)으로 설정했습니다.

### Data Augmentation

 Data augmentation은 2회차부터 넣었습니다. augmentation을 너무 많은 종류를 넣어도 성능에 영향을 줄 것이라고 생각해 pixel 단위에서 변환시키는augmentation 한 개와 이미지 자체에 변화를 주는 augmentation 한 개를 선택했습니다. augmentation은 train 과정에서만 적용해주었습니다.
Pixel 단위 augmentation은 ColorJitter transform을 사용했습니다. 이미지의 색과 빛의 세기에 관련 없이 모델이 분류를 잘 할 수 있을 것이라 예상했습니다. 이미지 자체에 변화를 주는 augmentation은 RandomHorizontalFlip을 사용했습니다. 이미지를 좌우 반전을 함으로써 분류 대상의 위치에 관련없이(해당 task에서는 입구의 위치에 관련 없이) 분류를 잘 할 것이라고 판단했습니다. 실제로 augmentation을 적용했을 때 성능이 1%가량 더 높게 나왔습니다.

### Result

[제목 없음](https://www.notion.so/fc45a25ba710414ab92235d78f4b14ae)

**>> validation accuracy**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b0e83f41-1c81-4839-ac93-a197d6040f00/Untitled.png)

 1회차 실험에서 accuracy가 심하게 변동이 있는 경향을 확인할 수 있었는데 learning rate를 조정한 뒤의 실험에서는 거의 일정하게 accuracy가 증가하는 그래프를 볼 수 있습니다.

**>> loss**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/615e8510-87d6-454a-bfec-42a6809039b7/Untitled.png)

  이번 task에서는 resnet18로 모델을 고정하고, hyper parameter의 조정과 image augmentation을 추가함으로써 성능을 향상시키는 것에 집중했습니다. Task에 맞는 augmentation을 예상해 적용시키고 hyper parameter를 맞춰준 결과 최대 99.12%의 성능을 만들었습니다. 다만 loss값이 감소할 때 심하게 튀는 issue가 있어 이 부분은 조금 더 생각해 볼 문제입니다.
