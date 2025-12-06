# バイキュービック補間（Bicubic Interpolation）

## 概要

バイキュービック補間は、画像処理において画像の拡大・縮小時に用いられる補間手法の一つである。隣接する16個のピクセル（4×4の範囲）を参照し、3次多項式（キュービック関数）を用いて新しいピクセル値を計算する手法である。

## 数学的定義

### キュービック関数

バイキュービック補間では、以下の3次多項式を使用する：

$$f(x) = \begin{cases}
(a+2)|x|^3 - (a+3)|x|^2 + 1 & \text{for } |x| \leq 1 \\
a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{for } 1 < |x| \leq 2 \\
0 & \text{for } |x| > 2
\end{cases}$$

ここで、パラメータ$a$は通常以下の値が使用される：
- $a = -0.5$（最も一般的）
- $a = -0.75$
- $a = -1.0$

### 2次元での実装

画像の場合、x方向とy方向の両方に対してキュービック補間を適用する：

1. まず、y方向に対して4つの列それぞれでキュービック補間を実行
2. 次に、得られた4つの値に対してx方向でキュービック補間を実行

$$I(x,y) = \sum_{i=0}^{3} \sum_{j=0}^{3} I_{i,j} \cdot f(x-i) \cdot f(y-j)$$

## アルゴリズム詳細

### ステップ1: 重み計算

目標位置$(x, y)$に対して、周囲16点の重みを計算：

```python
def cubic_weight(x, a=-0.5):
    """キュービック重み関数"""
    x = abs(x)
    if x <= 1:
        return (a + 2) * x**3 - (a + 3) * x**2 + 1
    elif x <= 2:
        return a * x**3 - 5*a * x**2 + 8*a * x - 4*a
    else:
        return 0

# x, y方向の重み計算
weights_x = [cubic_weight(x - i) for i in range(-1, 3)]
weights_y = [cubic_weight(y - j) for j in range(-1, 3)]
```

### ステップ2: 補間値計算

```python
def bicubic_interpolation(image, x, y, a=-0.5):
    """バイキュービック補間の実装"""
    # 整数部と小数部を分離
    x0, y0 = int(x), int(y)
    dx, dy = x - x0, y - y0
    
    # 周囲16点の重みを計算
    weights_x = [cubic_weight(dx - i, a) for i in range(-1, 3)]
    weights_y = [cubic_weight(dy - j, a) for j in range(-1, 3)]
    
    # 補間値を計算
    result = 0
    for j in range(4):
        for i in range(4):
            if 0 <= y0-1+j < image.shape[0] and 0 <= x0-1+i < image.shape[1]:
                pixel_value = image[y0-1+j, x0-1+i]
                result += pixel_value * weights_x[i] * weights_y[j]
    
    return result
```

## 特徴と性能

### 利点

1. **高品質な結果**: 最近傍補間やバイリニア補間と比較して、より滑らかで自然な結果
2. **エッジ保持**: エッジの鮮明さを比較的良く保持
3. **標準化**: 多くの画像処理ライブラリで標準実装

### 欠点

1. **計算コスト**: 16点を参照するため、計算量が多い
2. **オーバーシュート**: 場合によっては元の値域を超える値が生成される可能性
3. **境界処理**: 画像端での処理が複雑

### 計算量比較

| 手法 | 参照点数 | 計算量 | 品質 |
|------|----------|--------|------|
| 最近傍 | 1 | O(1) | 低 |
| バイリニア | 4 | O(1) | 中 |
| バイキュービック | 16 | O(1) | 高 |
| ランチョス | 可変 | O(n) | 最高 |

## 実用的な応用

### 画像リサイズ

```python
import cv2
import numpy as np

# OpenCVでのバイキュービック補間
resized = cv2.resize(image, (new_width, new_height), 
                    interpolation=cv2.INTER_CUBIC)
```

### 機械学習での前処理

```python
from torchvision.transforms import Resize
import torch.nn.functional as F

# PyTorchでのバイキュービック補間
transform = Resize((224, 224), interpolation=Image.BICUBIC)

# または直接的に
resized = F.interpolate(image_tensor, size=(224, 224), 
                       mode='bicubic', align_corners=False)
```

### 医療画像処理

- CTスキャンの3D再構成
- MRI画像の解像度向上
- X線画像の拡大表示

## パラメータ調整

### パラメータ$a$の影響

- $a = -0.5$: 標準的な設定、バランスの取れた結果
- $a = -0.75$: よりシャープな結果、エッジ強調
- $a = -1.0$: 最もシャープ、アーティファクトの可能性

### 品質評価指標

1. **PSNR (Peak Signal-to-Noise Ratio)**
2. **SSIM (Structural Similarity Index)**
3. **視覚的品質評価**

## 実装上の注意点

### 境界処理

```python
def safe_pixel_access(image, x, y):
    """境界を考慮した安全なピクセルアクセス"""
    h, w = image.shape[:2]
    x = max(0, min(w-1, x))
    y = max(0, min(h-1, y))
    return image[y, x]
```

### 数値安定性

- 浮動小数点演算の精度に注意
- オーバーフロー/アンダーフローの回避
- 適切なクランピング処理

## 最適化手法

### SIMD命令の活用

```cpp
// SSE/AVX命令を使った高速化例
__m128 weights = _mm_load_ps(weight_array);
__m128 pixels = _mm_load_ps(pixel_array);
__m128 result = _mm_mul_ps(weights, pixels);
```

### GPU実装

```cuda
__global__ void bicubic_kernel(float* input, float* output, 
                              int width, int height, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // GPU並列処理でのバイキュービック補間
}
```

## 具体的な数値例

### 例1: 2×2画像の拡大

元画像（2×2）:
```
100  200
150  250
```

この画像を4×4に拡大する場合を考える。位置(1.5, 1.5)での補間値を計算してみよう。

#### ステップ1: 重み計算

位置(1.5, 1.5)に対して、$a = -0.5$として重みを計算：

x方向の重み（dx = 0.5）:
- $f(-1.5) = 0$ (|x| > 2なので)
- $f(-0.5) = (-0.5+2)(0.5)^3 - (-0.5+3)(0.5)^2 + 1 = 1.5 \times 0.125 - 2.5 \times 0.25 + 1 = 0.5625$
- $f(0.5) = (-0.5+2)(0.5)^3 - (-0.5+3)(0.5)^2 + 1 = 0.5625$
- $f(1.5) = 0$ (|x| > 2なので)

y方向の重み（dy = 0.5）: 同様に計算

#### ステップ2: 補間値計算

境界処理を考慮した16点の値：
```
100  100  200  200   # 境界でパディング
100  100  200  200
150  150  250  250
150  150  250  250
```

補間値 = Σ(ピクセル値 × x重み × y重み)
= 100×0×0 + 100×0.5625×0 + 200×0.5625×0 + 200×0×0 + ...
= 175 (中央付近の値)

### 例2: より詳細な3×3例

元画像（3×3）:
```
 50  100  150
100  200  255
150  255  300
```

位置(1.0, 1.0)での補間値（ちょうど中央ピクセル）:

#### 重み計算（dx = 0, dy = 0）:
- $f(-1) = 0$
- $f(0) = 1$ (x=0の場合)
- $f(1) = 0$
- $f(2) = 0$

#### 補間値:
補間値 = 200×1×1 = 200 (元の値と一致)

### 例3: 非整数位置での補間

位置(1.25, 1.25)での補間値を計算：

#### 重み計算（dx = 0.25, dy = 0.25）:

$f(0.25)$の計算:
- $|x| = 0.25 \leq 1$なので第1式を使用
- $f(0.25) = (-0.5+2)(0.25)^3 - (-0.5+3)(0.25)^2 + 1$
- $= 1.5 \times 0.015625 - 2.5 \times 0.0625 + 1$
- $= 0.0234375 - 0.15625 + 1 = 0.8672$

$f(-0.75)$の計算:
- $|x| = 0.75 \leq 1$なので第1式を使用
- $f(0.75) = 1.5 \times (0.75)^3 - 2.5 \times (0.75)^2 + 1$
- $= 1.5 \times 0.4219 - 2.5 \times 0.5625 + 1 = 0.0391$

同様に他の重みも計算し、最終的な補間値を求める。

### 例4: 実際の計算確認

4×4の小さな画像での完全な計算例：

元画像:
```
 64   96  128  160
 96  128  160  192
128  160  192  224
160  192  224  256
```

位置(1.5, 1.5)での補間:

1. 周囲16点の抽出（境界処理込み）
2. x,y方向の重み計算
3. 重み付き平均の計算

最終結果: 約160（グラデーションの中央値）

## 参考文献

1. Keys, R. (1981). "Cubic convolution interpolation for digital image processing"
2. Parker, J.A. et al. (1983). "Comparison of interpolating methods for image resampling"
3. Lehmann, T.M. et al. (1999). "Survey: interpolation methods in medical image processing"

## 関連技術

- **スプライン補間**: より高次の多項式補間
- **ランチョス補間**: sinc関数ベースの補間
- **エッジ保持補間**: エッジを特別に処理する手法
- **学習ベース超解像**: 深層学習を用いた高品質補間