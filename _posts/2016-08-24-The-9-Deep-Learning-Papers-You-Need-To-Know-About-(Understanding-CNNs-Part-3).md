---
layout: post
title: 꼭 알아야 할 딥러닝 논문 9가지 (CNN 이해하기 3부)
author: 허정주
tags: translate-blog adeshpande3
subtitle: 딥러닝 논문 소개
category: translate-blog
---

원문: [바로가기](https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)

![표지](/assets/Cover3rd.png)

## 소개

[1부](https://iamroot-ai-2th.github.io/translate-blog/2016/07/20/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)

[2부](https://iamroot-ai-2th.github.io/translate-blog/2016/07/29/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/)

이 글에서 우리는 컴퓨터 비전과 콘볼루션 신경망 분야에서 많은 새롭고 중요한
발전을 요약할 것 입니다. 지난 5년동안 출판된 매우 중요한 논문 몇개를 보고 왜
그것이 중요한지 논할 것 입니다. AlexNet 부터 ResNet 까지 목록의 처음 절반은
일반적인 망 구조의 발전을 다루고, 나머지 절반은 다른 하위 영역의 흥미로운 논문의
모음입니다.

## [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (2012)

모든 시작은 하나입니다 (일부는 Yann LeCun 의 1998년
[논문](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)이 선구적인
출판이라고 말할 것 입니다). "ImageNet Classification with Deep Convolutional
Network" 라는 제목의 논문은 총 6,184 회 인용되고 이 분야에서 가장 영향력있는
출판물이라고 간주됩니다. 익숙하지 않은 사람을 위해 설명하자면 이 대회는 컴퓨터
비젼의 연간 올림픽이라고 간주할 수 있습니다. 누가 분류, 지역화, 탐지 등의 작업을
위한 더 좋은 컴퓨터 비젼 모델을 가지고 있는지 전세계 팀들이 경쟁하는 것을 볼 수
있는 곳입니다. 2012 는 15.4%의 상위 5가지 테스트 오류율을 달성하기 위해 CNN 이
사용된 첫해로 기록됩니다 (상위 5가지 오류는 모델이 주어진 이미지에 대해 예측한
것이 상위 5가지를 예측한 올바른 라벨과 맞지않는 확률입니다). 다음 최고 항목은
26.2% 의 오류를 달성했습니다. 이는 컴퓨터 비전 커뮤니티를 매우 놀라게 하는
충격적인 개선이었습니다. 그때부터 CNN 은 경쟁에서 누구나 아는 이름이
되었다고해도 과언이 아닙니다.

이 논문에서 그 그룹은 AlexNet 이라고 불리는 망의 구조를 논의했습니다. 그들은
현대 구조와 비교하면 상대적으로 간단한 배치를 사용했습니다. 망은 5개의
콘볼루션층, max-pooling 층, droupout 층과 3개의 완전히 연결된 층으로
구성되었습니다. 망은 1000 개의 범주를 분류하기 위해 설계되었습니다.

![AlexNet 그림](/assets/AlexNet.png)

### 주요 요점

* 전체 22,000 개 이상의 분류의 1,500 만개의 이미지를 포함하는 ImageNet 자료로
  망을 훈련하였습니다.
* 비선형 함수 ReLU 를 사용했습니다 (ReLU 가 기존 tanh 함수보다 몇배 빨라 훈련
  시간을 감소시켰습니다).
* 이미지 변환, 수평 반사, 부분 추출로 구성된 자료 보강 기법이 사용되었습니다.
* 훈련 자료에 과적합하는 문제에 대처하기 위해 드롭 아웃 층을 구현하였습니다.
* 가속도와 가중치 감소에 대한 특정 값으로 집단 확률 경사하강법을 사용하여 모델을
  훈련했습니다.
* 두개의 GTX 580 GPU 에서 **5-6일간** 훈련되었습니다.

### 중요한 이유

2012 년에 Krizhevsky, Sutckever, Hinton 에 의해 개발된 신경망은 컴퓨터 비젼
공동체에서 CNN 에 대한 발표였습니다. 이것은 역사적으로 어려운 ImageNet 데이터
세트에 잘 적용되는 첫 모델이었습니다. 자료 보강과 드롭 아웃같은 활용 기법은
현재에도 여전히 사용됩니다. 이 논문은 CNN 의 장점을 잘 설명하고 경쟁에서 기록을
깨는 성능으로 그것을 뒷받침합니다.

## [ZF Net](http://arxiv.org/pdf/1311.2901v3.pdf) (2013)

2012 년의 AlexNet 이 인기를 독차지 한 것에 힘입어, ILSVRC 2013 에 제출된 CNN
모델의 수가 급증하였습니다. 그 해 승자는 뉴욕대의 Matthew Zeiler 와 Rob Fergus
에 의해 만들어진 망이었습니다. ZF Net 이라는 이 모델은 11.2% 의 오류율을
달성하였습니다. 이 구조는 이전 AlexNet 구조를 약간 조정하였습니다. 그러나, 성능
개선에 대한 몇몇 핵심 발상이 개발되었습니다. 이것이 훌륭한 논문이라는 다른
이유는 저자들이 ConvNets 에 숨은 많은 직관을 설명하고 필터와 가중치를 어떻게
올바르게 시각화하는지 보여주는데 많은 시간을 보낸 것 입니다.

"Visualizing and Understanding Convolutional Neural Networks" 라는 이름의
논문에서, Zeiler 와 Fergus 는 CNN 의 새로운 관심은 대용량 훈련 세트의 접근성과
GPU 사용으로 증가된 연산 능력에 기인한다는 생각을 논의함으로써 시작하였습니다.
그들은 또한 연구자들이 이 모델의 내부 방법에서 경험한, 이 이해없이 "더 나은
모델의 개발은 시행착오로 몰아넣습니다" 라고 말하는, 제한된 지식에 대해
말하였습니다. 현재 3년전보다 더 나은 이해를 하지만, 많은 연구자에게 문제로
남아있습니다. 이 논문의 주요 기여는 조금 수정된 AlexNet 모델의 세부사항과 특징
지도를 시각화하는 매우 흥미로운 방법입니다.

![ZFNet 그림](/assets/zfnet.png)

### 주요 요점

* 몇가지 사소한 수정을 제외하면 AlexNet 과 매우 유사한 구조입니다.
* ZF Net 은 단지 130 만 이미지로 훈련된데반해, AlexNet 은 1,500 만 이미지로
  훈련되었습니다.
* 첫 층에 (AlexNet 이 사용한) 11x11 크기 필터를 사용하는 대신에, ZF Net 은 7x7
  크기의 필터를 사용하였고 이동 값을 줄였습니다. 이 변경의 배경는 첫 콘볼루션
  층의 작은 필터 크기가 입력 볼륨의 많은 원 화소 정보를 획득하는데 도움을 준다는
  것 입니다. 특히 첫 콘볼루션 층에서, 11x11 크기의 필터링이 많은 적절한 정보를
  생략하는 것이 증명되었습니다.
* 망이 성장함에 따라 사용된 필터의 수도 증가하였습니다.
* 활성화 함수에는 ReLU를 오류 함수에는 크로스엔트로피 손실을 사용하였고, 집단
  확률 경사하강법으로 훈련되었습니다.
* GTX 580 GPU 에서 **12일**간 훈련되었습니다.
* 디콘볼루션 망이라는 이름의 시각화 기술이 개발되었습니다. 이것은 특징 활성화와
  연관된 입력 공간의 다른점을 검사하는데 도움을 줍니다. 특징을 화소로 연관시키기
  때문에 "deconvnet" 이라 부릅니다 (콘볼루션 층이 하는 것의 반대).

## DeConvNet

훈련된 CNN 의 모든 층에서 이 작업을 수행하는 방법의 기본 개념은, 이미지 화소로
돌아가는 경로를 가지고 있는 "deconvnet" 을 붙이는 것 입니다. 입력 이미지가
CNN 에 주어지고 각 단계에서 활성화가 계산됩니다. 이것은 순방향 전달입니다. 이제,
네번째 콘볼루션층에서 구체적인 특징의 활성화를 검사한다고 가정해봅시다. 하나의
특징 지도의 활성화를 저장합니다. 그러나 이 층의 다른 활성화는 모두 0 으로
설정합니다. 그리고 이 특징 지도를 deconvnet 에 입력으로 전달합니다. 이
deconvnet 은 원 CNN 과 같은 필터를 가지고 있습니다. 이 입력은 입력 공간에 도달할
때 까지 선행층에 대한 역풀 (최대 풀링의 반대), 바로잡기, 필터 명령을 거칩니다.

이 모든 과정의 이유는 구조의 어떤 유형이 주어진 특징 지도를 활성화시키는지
검사하려는 것 입니다. 처음 두개 층을 시각적으로 살펴봅시다.

![ZFNet 그림 2](/assets/deconvnet.png)

[1부](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)
에서 논의 한 것 처럼, ConvNet 의 첫번째 층은 항상 저수준 특징을 찾아낼 것
입니다. 특별한 경우에 간단한 모서리나 색을 찾아낼 것 입니다. 두번쨰 층에서는
검출된 좀 더 반복적인특징을 볼 수 있습니다. 이제 3, 4, 5 층을 봅시다.

![ZFNet 그림 3](/assets/deconvnet2.png)

이 층들은 대의 얼굴이나 꽃 같은 고수준 특징을 더 많이 보여줍니다. 기억해야
할만한 한가지 참고사항은, 첫번째 층 이후, 일반적으로 이미지를 다운샘플링하는
풀링층을 가지고 있는 것 입니다. (예를 들어, 32x32x3 부피를 16x16x3 부피로
바꿉니다). 이것이 가지는 효과는 두번째 층이 원본 이미지에서 볼 수있는 폭
넓은 범위를 가지고 있는 것 입니다. deconvnet 이나 일반적인 논문에 대한 자세한
정보는, 주제에 관한 Zeiler 자신의
[발표](https://www.youtube.com/watch?v=ghEmQSxT6tw)를 확인하세요.

### 중요한 이유

ZF Net 은 2013 년의 경쟁의 승자일 뿐만 아니라, CNN 에서의 동작에 대한 훌륭한
직관을 제공하고 성능을 향상하기 위한 다양한 방법을 보여줬습니다. 설명된 시각화
접근법은 CNN 의 내부동작을 설명하는 것 뿐만 아니라, 망 구조 개선에 대한 통찰력을
제공하는데 도움을 줍니다. 흥미로운 deconv 시각화 접근법과 이에 대한 실험은
개인적으로 좋아하는 논문 중 하나입니다.

## [VGG Net](http://arxiv.org/pdf/1409.1556v6.pdf) (2014)

단순하고 깊게. 이것은 2014 년에 만들어지고 오류율이 7.3% 인 가장 잘 활용한
모델입니다. 옥스포드 대학의 Karen Simonyan 과 Andrew Zisserman 이 19 층의 CNN 을
만들었습니다. 스트라이드와 패드가 1 인 3x3 필터와, 스트라이드가 2 인 2x2 최대
풀링을 엄격하게 사용하였습니다. 충분히 간단하죠?

![VGGNet 그림](/assets/VGGNet.png)

### 주요 요점

* 첫번째 층에서 3x3 크기의 필터만 사용한 것은 AlexNet 의 11x11 필터 및 ZF Net 의
  7x7 필터와 상당히 다릅니다. 저자의 논리는 두개의 3x3 콘볼루션 층의 조합은
  효과적으로 받아들이는 5x5 의 영역을 가지고 있다는 것 입니다. 이것은 작은 필터
  크기의 장점을 유지하면서 큰 필터를 시뮬레이션합니다. 장점 중 하나는 매개변수의
  개수가 감소하는 것 입니다. 또한, 2 개의 콘볼루션 층을 사용하면, 2 개의
  ReLU 층을 사용할 수 있습니다.
* 3 개의 콘볼루션 층은 7x7 과 같은 효과를 냅니다.
* 각 층에서 입력 볼륨의 공간 크기 감소로 (콘볼루션과 풀 층의 결과), 증가된 필터
  개수에 따라 볼륨의 깊이가 증가합니다.
* 흥미로운 점은 각 최대 풀 층 이후 필터의 수가 두배가 됩니다. 이것은 공간 차원
  축소 발상을 강화하지만, 깊이를 증가시킵니다.
* 이미지 분류와 로컬라이제이션 작업 모두 잘 동작합니다. 저자는 회귀방법으로
  로컬라이제이션의 형식을 사용하였습니다 (모든 자세한 정보는
  [논문](http://arxiv.org/pdf/1409.1556v6.pdf)의 10 쪽을 보세요).
* Caffe 툴박스로 만들어진 모델입니다.
* 훈련기간동안 하나의 자료 보강 기법으로 스케일 지터링을 사용하였습니다.
* 각 콘볼루션 층 다음에 ReLU 층을 사용하였고 경사 하강으로 훈련되었습니다.
* 4 개의 Nvidia Titan Black GPU 에서 **2 ~ 3 주**동안 훈련되었습니다.

### 중요한 이유

VGG Net 은 다음 개념을 보강하였기 때문에 나에게 많은 영향을 준 논문중
하나입니다. **콘볼루션 신경망이 시각 자료의 계층적 표현에 대해 동작하려면 층의
깊은 망을 가지고 있어야 합니다.** 깊게. 간단하게.

## [GoogLeNet](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) (2015)

우리가 얘기하는 망 설계에서 단순성의 개념을 알고있나요? 구글은 인셉션 모델의
소개와 함께 그것을 보여주었습니다. GoogLeNet 은 22 층의 CNN 이고 상위 5가지
오류율이 6.7% 로 ILSVRC 2014 의 우승자입니다. 제가 알기로, 이것은 순차적
구조에서 서로의 상단에 콘볼루션과 풀링 층을 쌓는 일반적인 접근에서 벗어난 첫 CNN
구조중 하나입니다. 논문의 저자는 이 새로운 모델은 메모리와 전력 사용에 있어
중요한 고려 사항이 있다고 강조합니다. (가끔 잊어버리는 중요한 참고: 이모든 층을
쌓고 많은 필터를 추가하는 것은 연산과 메모리 비용뿐만 아니라 과적합 될 가능성이
높아집니다).

![GoogleNet](/assets/GoogleNet.gif)

![GoogleNet 그림](/assets/GoogLeNet.png)

### 인셉션 모듈

처음 GoogLeNet 의 구조를 접했을 때, 이전 구조에서 본 것 처럼 모두 순차적으로
진행되지 않는 것을 즉시 알아차릴 것입니다. 병렬로 진행되는 망의 조각이 있습니다.

![GoogleNet 그림 2](/assets/GoogLeNet2.png)

이 상자는 인셉션 모듈이라 불립니다. 어떻게 구성되어 있는지 자세히 봅시다.

![GoogleNet 그림 3](/assets/GoogLeNet3.png)

아래 초록 상자는 우리의 입력이며 위의 것은 모델의 출력입니다 (이 사진을
오른쪽으로 90도 돌리면 전체 망을 보여주는 마지막 그림과 관계하여 모델을 그려볼
수 있습니다). 기본적으로, 전통적인 ConvNet 의 각 층에서, 풀링 연산을 할지
콘볼루션 연산을 할지 여부를 선택해야 합니다 (필터 크기의 선택도 있습니다).
인셉션 모듈은 이러한 연산을 모두 병렬로 수행할 수 있게 해줍니다. 사실, 이것은
저자가 찾아낸 정확히 "na&iuml;ve" 한 생각입니다.

![GoogLeNet 그림 4](/assets/GoogLeNet4.png)

자, 이것이 왜 작동하지 않을까요? 그것은 너무 많은 출력으로 **이어질** 것 입니다.
우리는 출력 볼륨에 대한 매우 크고 깊은 채널로 끝날 것 입니다. 저자가 이 문제를
해결한 방법은 3x3 과 5x5 층 전에 1x1 콘볼루션 연산을 추가하는 것입니다. 1x1
콘볼루션 (또는 망 층의 망) 은 차원 축소하는 방법을 제공합니다. 예를 들어,
100x100x60 의 입력 볼륨이 있다고 가정합시다 (이미지의 면적은 중요하지 않습니다.
단지 망의 아무 층의 입력일 뿐 입니다). 1x1 콘볼루션 필터 20 개를 적용하면 볼륨을
100x100x20 으로 줄일 수 있습니다. 이것은 3x3 과 5x5 콘볼루션이 처리해야할 볼륨이
크지 않다는 의미입니다. 볼륨의 깊이를 줄였기때문에, 이것은 "특징의 풀링"으로
생각될 수 있습니다. 보통 최대 풀링 층으로 높이와 가로폭의 차원을 줄인 방법과
유사합니다. 다른 참고사항은 다음에 ReLU 유닛이 오는 이 1x1 콘볼루션 층은 절대
다칠 수 없습니다. (1x1 콘볼루션의 효과상에 대한 자세한 정보는 Aaditya Prakash 의
[훌륭한 글](http://iamaaditya.github.io/2016/03/one-by-one-convolution/)을
보세요). 끝에 연결된 필터의 훌륭한 시각화에 대한 이
[영상](https://www.youtube.com/watch?v=VxhSouuSZDY)을 확인하세요.

You may be asking yourself "How does this architecture help?". Well, you have a
module that consists of a network in network layer, a medium sized filter
convolution, a large sized filter convolution, and a pooling operation. The
network in network conv is able to extract information about the very fine grain
details in the volume, while the 5x5 filter is able to cover a large receptive
field of the input, and thus able to extract its information as well. You also
have a pooling operation that helps to reduce spatial sizes and combat
overfitting. On top of all of that, you have ReLUs after each conv layer, which
help improve the nonlinearity of the network. Basically, the network is able to
perform the functions of these different operations while still remaining
computationally considerate. The paper does also give more of a high level
reasoning that involves topics like sparsity and dense connections (read
Sections 3 and 4 of the
[paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf).
Still not totally clear to me, but if anybody has any insights, I'd love to hear
them in the comments!).

### 주요 요점

* Used 9 Inception modules in the whole architecture, with over 100 layers in
  total! Now that is deep...
* No use of fully connected layers! They use an average pool instead, to go from
  a 7x7x1024 volume to a 1x1x1024 volume. This saves a huge number of
  parameters.
* Uses 12x fewer parameters than AlexNet.
* During testing, multiple crops of the same image were created, fed into the
  network, and the softmax probabilities were averaged to give us the final
  solution.
* Utilized concepts from R-CNN (a paper we'll discuss later) for their detection
  model.
* There are updated versions to the Inception module (Versions 6 and 7).
* Trained on "a few high-end GPUs **within a week**".

### 중요한 이유

GoogLeNet was one of the first models that introduced the idea that CNN layers
didn't always have to be stacked up sequentially. Coming up with the Inception
module, the authors showed that a creative structuring of layers can lead to
improved performance and computationally efficiency. This paper has really set
the stage for some amazing architectures that we could see in the coming years.

![GoogleNet pic 5](/assets/GoogLeNet5.png)

## [Microsoft ResNet](https://arxiv.org/pdf/1512.03385v1.pdf) (2015)

Imagine a deep CNN architecture. Take that, double the number of layers, add a
couple more, and it still probably isn't as deep as the ResNet architecture that
Microsoft Research Asia came up with in late 2015. ResNet is a new 152 layer
network architecture that set new records in classification, detection, and
localization through one incredible architecture. Aside from the new record in
terms of number of layers, ResNet won ILSVRC 2015 with an incredible error rate
of 3.6% (Depending on their skill and expertise, humans generally hover around a
5-10% error rate. See Andrej Karpathy's
[great post](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/)
on his experiences with competing against ConvNets on the ImageNet challenge).

![ResNet](/assets/ResNet.gif)

### Residual Block

The idea behind a residual block is that you have your input x go through
conv-relu-conv series. This will give you some F(x). That result is then added
to the original input x. Let's call that H(x) = F(x) + x. In traditional CNNs,
your H(x) would just be equal to F(x) right? So, instead of just computing that
transformation (straight from x to F(x)), we're computing the term that you have
to *add*, F(x), to your input, x. Basically, the mini module shown below is
computing a "delta" or a slight change to the original input x to get a slightly
altered representation (When we think of traditional CNNs, we go from x to F(x)
which is a completely new representation that doesn't keep any information about
the original x). The authors believe that "it is easier to optimize the residual
mapping than to optimize the original, unreferenced mapping".

![ResNet 그림](/assets/ResNet.png)

Another reason for why this residual block might be effective is that during the
backward pass of backpropagation, the gradient will flow easily through the
effective because we have addition operations, which distributes the gradient.

### 주요 요점

* "Ultra-deep" &ndash; Yann LeCun.
* 152 layers&hellip;
* Interesting note that after only the *first 2* layers, the spatial size gets
  compressed from an input volume of 224x224 to a 56x56 volume.
* Authors claim that a na&iuml;ve increase of layers in plain nets result in
  higher training and test error (Figure 1 in the
  [paper](https://arxiv.org/pdf/1512.03385v1.pdf)).
* The group tried a 1202-layer network, but got a lower test accuracy,
  presumably due to overfitting.
* Trained on an 8 GPU machine for **two to three weeks**.

### 중요한 이유

3.6% error rate. That itself should be enough to convince you. The ResNet model
is the best CNN architecture that we currently have and is a great innovation
for the idea of residual learning. With error rates dropping every year since
2012, I'm skeptical about whether or not they will go down for ILSVRC 2016. I
believe we've gotten to the point where stacking more layers on top of each
other isn't going to result in a substantial performance boost. There would
definitely have to be creative new architectures like we've seen the last 2
years. On September 16<sup>th</sup>, the results for this year's competition
will be released. Mark your calendar.

**Bonus**: [ResNets inside of ResNets](http://arxiv.org/pdf/1608.02908.pdf).
Yeah. I went there.

## Region Based CNNs ([R-CNN](https://arxiv.org/pdf/1311.2524v5.pdf) - 2013, [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf) - 2015, [Faster R-CNN](http://arxiv.org/pdf/1506.01497v3.pdf) - 2015)

Some may argue that the advent of R-CNNs has been more impactful that any of the
previous papers on new network architectures. With the first R-CNN paper being
cited over 1600 times, Ross Girshick and his group at UC Berkeley created one of
the most impactful advancements in computer vision. As evident by their titles,
Fast R-CNN and Faster R-CNN worked to make the model faster and better suited
for modern object detection tasks.

The purpose of R-CNNs is to solve the problem of object detection. Given a
certain image, we want to be able to draw bounding boxes over all of the
objects. The process can be split into two general components, the region
proposal step and the classification step.

The authors note that any class agnostic region proposal method should fit.
[Selective Search](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf)
is used in particular for RCNN. Selective Search performs the function of
generating 2000 different regions that have the highest probability of
containing an object. After we've come up with a set of region proposals, these
proposals are then "warped" into an image size that can be fed into a trained
CNN (AlexNet in this case) that extracts a feature vector for each region. This
vector is then used as the input to a set of linear SVMs that are trained for
each class and output a classification. The vector also gets fed into a bounding
box regressor to obtain the most accurate coordinates.

![RCNN 그림](/assets/rcnn.png)

Non-maxima suppression is then used to suppress bounding boxes that have a
significant overlap with each other.

### Fast R-CNN

Improvements were made to the original model because of 3 main problems.
Training took multiple stages (ConvNets to SVMs to bounding box regressors), was
computationally expensive, and was extremely slow (RCNN took 53 seconds per
image). Fast R-CNN was able to solve the problem of speed by basically sharing
computation of the conv layers between different proposals and swapping the
order of generating region proposals and running the CNN. In this model, the
image is *first *fed through a ConvNet, features of the region proposals are
obtained from the last feature map of the ConvNet (check section 2.1 of the
[paper](https://arxiv.org/pdf/1504.08083.pdf) for more details), and lastly we
have our fully connected layers as well as our regression and classification
heads.

![Fast RCNN 그림](/assets/FastRCNN.png)

### Faster R-CNN

Faster R-CNN works to combat the somewhat complex training pipeline that both
R-CNN and Fast R-CNN exhibited. The authors insert a region proposal network
(RPN) after the last convolutional layer. This network is able to just look at
the last convolutional feature map and produce region proposals from that. From
that stage, the same pipeline as R-CNN is used (ROI pooling, FC, and then
classification and regression heads).

![Faster RCNN pic](/assets/FasterRCNN.png)

### 중요한 이유

Being able to determine that a specific object is in an image is one thing, but
being able to determine that object's exact location is a huge jump in knowledge
for the computer. Faster R-CNN has become the standard for object detection
programs today.

## [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf) (2014)

[According to Yann LeCun](https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning),
these networks could be the next big development. Before talking about this
paper, let's talk a little about adversarial examples. For example, let's
consider a trained CNN that works well on ImageNet data. Let's take an example
image and apply a perturbation, or a slight modification, so that the prediction
error is *maximized*. Thus, the object category of the prediction changes, while
the image itself looks the same when compared to the image without the
perturbation. From the highest level, adversarial examples are basically
the images that fool ConvNets.

![GAN 그림](/assets/Adversarial.png)

Adversarial examples ([paper](http://arxiv.org/pdf/1312.6199v4.pdf)) definitely
surprised a lot of researchers and quickly became a topic of interest. Now let's
talk about the generative adversarial networks. Let's think of two models, a
generative model and a discriminative model. The discriminative model has the
task of determining whether a given image looks natural (an image from the
dataset) or looks like it has been artificially created. The task of the
generator is to create images so that the discriminator gets trained to produce
the correct outputs. This can be thought of as a zero-sum or minimax two player
game. The analogy used in the paper is that the generative model is like "a team
of counterfeiters, trying to produce and use fake currency" while the
discriminative model is like "the police, trying to detect the counterfeit
currency". The generator is trying to fool the discriminator while the
discriminator is trying to not get fooled by the generator. As the models train,
both methods are improved until a point where the "counterfeits are
indistinguishable from the genuine articles".

### 중요한 이유

Sounds simple enough, but why do we care about these networks? As Yann LeCun
stated in his Quora
[post](https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning),
the discriminator now is aware of the "internal representation of the data"
because it has been trained to understand the differences between real images
from the dataset and artificially created ones. Thus, it can be used as a
feature extractor that you can use in a CNN. Plus, you can just create really
cool artificial images that look pretty natural to me
([link](http://soumith.ch/eyescream/)).

## [Generating Image Descriptions](https://arxiv.org/pdf/1412.2306v2.pdf) (2014)

What happens when you combine CNNs with RNNs (No, you don't get R-CNNs, sorry)?
But you do get one really amazing application. Written by Andrej Karpathy (one
of my personal favorite authors) and Fei-Fei Li, this paper looks into a
combination of CNNs and bidirectional RNNs (Recurrent Neural Networks) to
generate natural language descriptions of different image regions. Basically,
the model is able to take in an image, and output this:

![Image Description 그림](/assets/Caption.png)

That's pretty incredible. Let's look at how this compares to normal CNNs. With
traditional CNNs, there is a single clear label associated with each image in
the training data. The model described in the paper has training examples that
have a sentence (or caption) associated with each image. This type of label is
called a weak label, where segments of the sentence refer to (unknown) parts of
the image. Using this training data, a deep neural network "infers the latent
alignment between segments of the sentences and the region that they describe"
(quote from the paper). Another neural net takes in the image as input and
generates a description in text. Let's take a separate look at the two
components, alignment and generation.

### Alignment Model

The goal of this part of the model is to be able to align the visual and textual
data (the image and its sentence description). The model works by accepting an
image and a sentence as input, where the output is a score for how well they
match (Now, Karpathy refers a different
[paper](https://arxiv.org/pdf/1406.5679v1.pdf) which goes into the specifics of
how this works. This model is trained on compatible and incompatible
image-sentence pairs).

Now let's think about representing the images. The first step is feeding the
image into an R-CNN in order to detect the individual objects. This R-CNN was
trained on ImageNet data. The top 19 (plus the original image) object regions
are embedded to a 500 dimensional space. Now we have 20 different 500
dimensional vectors (represented by v in the paper) for each image. We have
information about the image. Now, we want information about the sentence. We're
going to embed words into this same multimodal space. This is done by using a
bidirectional recurrent neural network. From the highest level, this serves to
illustrate information about the context of words in a given sentence. Since
this information about the picture and the sentence are both in the same space,
we can compute inner products to show a measure of similarity.

### Generation Model

The alignment model has the main purpose of creating a dataset where you have a
set of image regions (found by the RCNN) and corresponding text (thanks to the
BRNN). Now, the generation model is going to learn from that dataset in order to
generate descriptions given an image. The model takes in an image and feeds it
through a CNN. The softmax layer is disregarded as the outputs of the fully
connected layer become the inputs to another RNN. For those that aren't as
familiar with RNNs, their function is to basically form probability
distributions on the different words in a sentence (RNNs also need to be trained
just like CNNs do).

**Disclaimer:** This was definitely one of the more dense papers in this
section, so if anyone has any corrections or other explanations, I'd love to
hear them in the comments!

![Image Description 그림 2](/assets/GeneratingImageDescriptions.png)

### 중요한 이유

The interesting idea for me was that of using these seemingly different RNN and
CNN models to create a very useful application that in a way combines the fields
of Computer Vision and Natural Language Processing. It opens the door for new
ideas in terms of how to make computers and models smarter when dealing with
tasks that cross different fields.

## [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf) (2015)

Last, but not least, let's get into one of the more recent papers in the field.
This paper was written by a group at Google Deepmind a little over a year ago.
The main contribution is the introduction of a Spatial Transformer module. The
basic idea is that this module transforms the input image in a way so that the
subsequent layers have an easier time making a classification. Instead of making
changes to the main CNN architecture itself, the authors worry about making
changes to the image *before *it is fed into the specific conv layer. The 2
things that this module hopes to correct are pose normalization (scenarios where
the object is tilted or scaled) and spatial attention (bringing attention to the
correct object in a crowded image). For traditional CNNs, if you wanted to make
your model invariant to images with different scales and rotations, you'd need a
lot of training examples for the model to learn properly. Let's get into the
specifics of how this transformer module helps combat that problem.

The entity in traditional CNN models that dealt with spatial invariance was the
maxpooling layer. The intuitive reasoning behind this later was that once we
know that a specific feature is in the original input volume (wherever there are
high activation values), it's exact location is not as important as its relative
location to other features. This new spatial transformer is dynamic in a way
that it will produce different behavior (different distortions/transformations)
for each input image. It's not just as simple and pre-defined as a traditional
maxpool. Let's take look at how this transformer module works. The module
consists of:

* A localization network which takes in the input volume and outputs parameters
  of the spatial transformation that should be applied. The parameters, or
  theta, can be 6 dimensional for an affine transformation.
* The creation of a sampling grid that is the result of warping the regular grid
 with the affine transformation (theta) created in the localization network.
* A sampler whose purpose is to perform a warping of the input feature map.

![STN 그림](/assets/SpatialTransformer.png)

This module can be dropped into a CNN at any point and basically helps the
network learn how to transform feature maps in a way that minimizes the cost
function during training.

![STN 그림 2](/assets/SpatialTransformer2.png)

### 중요한 이유

This paper caught my eye for the main reason that improvements in CNNs don't
necessarily have to come from drastic changes in network architecture. We don't
need to create the next ResNet or Inception module. This paper implements the
simple idea of making affine transformations to the input image in order to help
models become more invariant to translation, scale, and rotation. For those
interested, here is a
[video](https://drive.google.com/file/d/0B1nQa_sA3W2iN3RQLXVFRkNXN0k/view) from
Deepmind that has a great animation of the results of placing a Spatial
Transformer module in a CNN and a good Quora
[discussion](https://www.quora.com/How-do-spatial-transformer-networks-work).

And that ends our 3 part series on ConvNets! Hope everyone was able to follow
along, and if you feel that I may have left something important out,
**let me know in the comments**! If you want more info on some of these
concepts, I once again highly recommend Stanford CS 231n lecture videos which
can be found with a simple YouTube search.

[Sources](/assets/Sources3.txt)
