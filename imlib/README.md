# imlib2

이전까지 사용했던 imlib를 내용적으로 정리하기 위해 class를 이용해서 다시 작성하였다. 크게 base.py를 통해 ImageBase class를 생성하고 여기서 IO 및 기본적인 plot function을 구현하였다. 그 다음으로 이를 기반으로 해서 base_filters.py에서 filter에 관한 함수 들을 추가 하였다. 여기에는 주로 이미지 자체를 바꾸는 여러 filter들을 주로 구현하였다. 그리고 다음으로 base_lines.py에서 line profile에 관련된 함수들을 추가하였다. 여기에서는 이미지 자체가 변하는 process는 없다. 하지만 주로 3d mean project 된 이미지를 가지고 작업하는 함수가 주로이다. 마지막으로 이 모든 것을 포함한 base_features.py에서는 pattern recognition에 관한 함수들을 구현하였다.

## Install

다음 패키지가 필요하다.

- TifFile: tifffile, numpy, scipy, scikit-image, matplotlib, lmfit
- SorterTif: matplotlib, numpy, scipyt, lmfit
- TrackerTif: matplitlib, numpy, scipy, scikit-image

## Initialization

\__init__.py 파일을 통해 다음의 class 를 로딩한다.
from imlib2.base import ImageBase
from imlib2.base_filters import ImageFilter
from imlib2.base_lines import ImageLine
from imlib2.base_features import ImageFeature
from imlib2.imgfolder import ImgFolder

사실 ImgFolder를 로딩하면 다른 클래스들은 자동으로 포함되게 된다.

[TODO] loading에 시간이 많이 걸리는데, 이유를 찾아보자. 규모가 큰 package들은 그 내용이 사용될때 import하도록 하자. skimage, pandas, cv2, trackpy, openpiv 등이 그러한 패키지 들이다.

## Naming Rules

함수가 여럿이기 때문에 정확한 규칙이 없으면 사용하기가 쉽지 않다. 그래서 다음의 규칙을 정하고 가능한 한 지키려고 하였다.

- underscore(_)로 시작되는 변수들은 class 내부 변수들이다. 주로 내부적으로 공유되거나 저장되어야 할 필요가 있는 변수들이다.
- underscore(_)로 시작되는 method들은 class 내부에 있다면 보조적으로 쓰이는 함수 이거나 frame number를 사용하지 않고 image 를 직접 입력으로 받는 함수 들이다. 예를 들어 \_show는 show에 보조되는 함수로 show(frame, \**kwargs)처럼 사용될 때 \_show(img, \**kwargs)로 사용이 된다.
- 함수 이름에 frame, line 등이 들어가면 하나의 image frame 혹은 하나의 line에 적용되는 함수 이고 frames, lines 처럼 복수로 사용되면 그 함수는 다수의 입력 혹은 label들을 입력으로 받는 함수 이다.
- class 밖에 정의되는 함수들은 underscore(_)로 시작되며 보조적인 함수로 쓰인다. 가능하면 @numba.jit(nopython=True)가 되도록 작성하자. numba가 작동하기 위해서는 numpy와 기본 python 만으로 된 함수이면 된다. 또한 어짜피 c로 compile이 될 것이기 때문에, for loop를 사용하는 것도 괜찮다.

## ImageBase

가장 신경을 많이 써야 하는 기본 클래스이다. 또한 최적화에 신경을 써야 한다. 꼭 필요한 패키지만 로드하도록 하자. File IO를 위해 pims와 tifffile를 사용하였다.

- 기왕이면 각 패키지에서 사용되는 함수는 하나 하나 따로 써 주도록 하자. 효율이 높지는 않지만, 어떤 함수들이 사용되었는지 바로 알 수 있다.
- 그리고 \__all__를 정해서 어떤 함수들이 외부에서 사용되는지 알리도록 하자.
