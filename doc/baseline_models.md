# ベースラインモデル: 段階的デバッグアプローチ

## 概要

学習がうまくいかない問題を切り分けるため、複雑度の異なる4つのベースラインモデルを実装しました。
すべて**事前学習済みResNet18**をベースにしており、ImageNetの知識を活用できます。

## モデル一覧

### 1. ResNet18_SimplerFCN (11.5M parameters)
**最もシンプル** - FCNスタイル

```python
from src_inference1.ResNet18_Simple import ResNet18_SimplerFCN

model = ResNet18_SimplerFCN(pretrained=True, num_classes=1)
```

**特徴:**
- エンコーダーのみ (ResNet18)
- 1x1畳み込みでチャンネル削減
- バイリニア補間によるアップサンプリング
- U-Net、LSTM、FPN なし

**用途:** データとトレーニングループが正常かを確認

---

### 2. ResNet18_Simple (14M parameters)
転置畳み込みデコーダー

```python
from src_inference1.ResNet18_Simple import ResNet18_Simple

model = ResNet18_Simple(pretrained=True, num_classes=1)
```

**特徴:**
- ResNet18エンコーダー
- 転置畳み込みによる学習可能なデコーダー
- U-Net、LSTM、FPN なし

**用途:** デコーダーの学習能力を確認

---

### 3. ResNet18_U (13M parameters)
U-Net構造（時系列なし）

```python
from src_inference1.ResNet18_U import ResNet18_U

model = ResNet18_U(pretrained=True, num_classes=1, ps=0.0)
```

**特徴:**
- ResNet18エンコーダー
- U-Netスタイルのスキップ接続
- FPN (Feature Pyramid Network)
- LSTMなし（単一フレームまたは最終フレームのみ使用）

**用途:** スキップ接続とFPNの効果を確認

---

### 4. ResNet18_ULSTM (15M parameters)
完全版（時系列モデリング付き）

```python
from src_inference1.ResNet18_ULSTM import ResNet18_ULSTM

model = ResNet18_ULSTM(pretrained=True, num_classes=1, ps=0.0)
```

**特徴:**
- ResNet18エンコーダー
- LSTM（深い2層で時系列モデリング）
- U-Net + FPN
- **CoaT_ULSTMと同じ構造**（エンコーダーのみ異なる）

**用途:** 時系列モデリングの効果を確認

---

## 使用方法

### 基本的な推論

```python
import torch
from src_inference1.ResNet18_U import ResNet18_U

# モデル初期化（事前学習済み重み自動ダウンロード）
model = ResNet18_U(pretrained=True)
model.eval()

# 入力データ
# 時系列: (B, C, T, H, W) = (batch, 3, 5, 256, 256)
# または単一フレーム: (B, C, H, W) = (batch, 3, 256, 256)
x = torch.randn(1, 3, 5, 256, 256)

# 推論
with torch.no_grad():
    output = model(x)  # (1, 1, 256, 256)
    pred = torch.sigmoid(output)
```

### トレーニングでの使用

```python
from src_inference1.ResNet18_U import ResNet18_U
import torch.optim as optim

# モデル初期化
model = ResNet18_U(pretrained=True, ps=0.1)  # Dropout 10%

# オプティマイザ
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# トレーニングループ
for epoch in range(epochs):
    for imgs, masks in train_loader:
        optimizer.zero_grad()
        
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
```

## 推奨テスト順序

### Phase 1: 最小構成で学習可能性を確認

```python
from src_inference1.ResNet18_Simple import ResNet18_SimplerFCN

model = ResNet18_SimplerFCN(pretrained=True)

# 推奨設定:
# - size=100 (データ数を制限)
# - epochs=1-2
# - LR=1e-3 (事前学習済みなので低めでOK)
# - BS=8, GradientAccumulation=4
# - FP32 (FP16無効化)
```

**期待される結果:**
- ✅ Loss < 0.5 (epoch 1終了時)
- ✅ NaN が発生しない
- ✅ F1 > 0.2 (最低限の学習)

**失敗した場合:** データ前処理またはトレーニングループに問題あり

---

### Phase 2: U-Net構造で精度向上

```python
from src_inference1.ResNet18_U import ResNet18_U

model = ResNet18_U(pretrained=True, ps=0.0)

# 同じ設定で学習
```

**期待される結果:**
- ✅ Loss < 0.4
- ✅ F1 > 0.4
- ✅ Phase 1より精度向上

---

### Phase 3: 時系列モデリング追加

```python
from src_inference1.ResNet18_ULSTM import ResNet18_ULSTM

model = ResNet18_ULSTM(pretrained=True, ps=0.0)
```

**期待される結果:**
- ✅ Loss < 0.35
- ✅ F1 > 0.5
- ✅ 時系列情報で精度向上

---

### Phase 4: CoaTエンコーダーに置き換え

```python
from src_inference1.CoaT_ULSTM import CoaT_ULSTM

# 事前学習済み重みをダウンロード
# https://github.com/mlpc-ucsd/CoaT
model = CoaT_ULSTM(pre="path/to/coat_lite_medium.pth", arch="medium")
```

**期待される結果:**
- ✅ ResNet18_ULSTMと同等以上の精度
- ✅ より深いネットワークで表現力向上

---

## モデル比較表

| モデル | パラメータ数 | 事前学習 | LSTM | U-Net | FPN | 用途 |
|--------|-------------|----------|------|-------|-----|------|
| ResNet18_SimplerFCN | 11.5M | ✅ ImageNet | ❌ | ❌ | ❌ | 最小構成テスト |
| ResNet18_Simple | 14M | ✅ ImageNet | ❌ | ❌ | ❌ | デコーダーテスト |
| ResNet18_U | 13M | ✅ ImageNet | ❌ | ✅ | ✅ | スキップ接続テスト |
| ResNet18_ULSTM | 15M | ✅ ImageNet | ✅ | ✅ | ✅ | 時系列テスト |
| CoaT_ULSTM | 35M | ✅ ImageNet | ✅ | ✅ | ✅ | 最終モデル |
| BaseCoatULSTM | 35M | ❌ なし | ✅ | ✅ | ✅ | ⚠️ 学習困難 |

---

## トラブルシューティング

### NaN Loss が発生する場合

1. **FP16を無効化**
   ```python
   learn = Learner(...).to_fp32()
   ```

2. **学習率を下げる**
   ```python
   lr_max = 1e-4  # 1e-3から下げる
   ```

3. **Gradient Clippingを追加**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

4. **Lovasz損失にepsilonを追加**
   ```python
   # src_inference1/lovasz.py
   jaccard = 1. - intersection / (union + 1e-7)
   ```

### 学習が遅い場合

1. **バッチサイズを増やす**
   ```python
   BS = 16  # 2から増やす
   GradientAccumulation(2)  # 16から減らす
   ```

2. **DataLoaderのnum_workersを増やす**
   ```python
   DataLoader(..., num_workers=4)
   ```

3. **混合精度を有効化**（NaNが出ない場合）
   ```python
   learn = Learner(...).to_fp16()
   ```

### 精度が上がらない場合

1. **学習率スケジュールを調整**
   ```python
   # OneCycleLR
   pct_start = 0.3  # 0.1から増やす
   ```

2. **データ拡張を強化**
   ```python
   from src_inference1.data import get_aug
   ds_train = ContrailsDatasetV0(..., tfms=get_aug())
   ```

3. **Dropoutを追加**
   ```python
   model = ResNet18_U(pretrained=True, ps=0.1)
   ```

---

## まとめ

このベースラインモデル群により:

✅ **問題の切り分けが可能** - どの段階で問題が発生するか特定できる
✅ **段階的な複雑化** - シンプルなモデルから徐々に機能を追加
✅ **事前学習の活用** - ImageNetの知識で学習を安定化
✅ **高速な検証** - 軽量モデルで素早く実験

次のステップ:
1. `eda/base_train.ipynb`で各モデルの推論を比較
2. 最もシンプルな`ResNet18_SimplerFCN`から学習開始
3. 段階的に複雑度を上げて精度向上
4. 最終的に`CoaT_ULSTM`で最高精度を目指す
