# Qwen2.5-VL JavaScript 实现

纯 JavaScript 实现的 Qwen2.5-VL-3B 视觉语言模型推理，完全摆脱 Python 依赖。

## 快速开始

```bash
# 安装依赖
pnpm i

# 运行推理
node run.js
```

## 功能特性

- ✅ 完整的图像预处理（smart_resize、patch extraction、normalize）
- ✅ 端到端的 JavaScript 推理
- ✅ 与 Python 版本输出完全一致
- ✅ 性能相当甚至略优（~1.1 tok/s vs ~1.0 tok/s）
- ✅ 更低的内存占用

## 技术架构

```
JavaScript Runtime
    ├── @huggingface/transformers
    │   ├── RawImage (图像读取和 resize)
    │   └── AutoTokenizer (文本 tokenization)
    ├── onnxruntime-node
    │   ├── vision_encoder (图像特征提取)
    │   ├── embed_tokens (文本嵌入)
    │   └── decoder_model (生成解码)
    └── 原生 JS
        ├── 图像预处理 (smart_resize, patch extraction, normalize)
        ├── 数值计算 (RoPE, 位置编码)
        └── Token 构建和管理
```

## 核心实现

### 1. 图像预处理

完全用 JavaScript 实现 Python processor 的图像处理逻辑：

```javascript
// smart_resize 算法
const factor = patchSize * mergeSize;
let hBar = Math.round(origHeight / factor) * factor;
let wBar = Math.round(origWidth / factor) * factor;

if (hBar * wBar > maxPixels) {
  const beta = Math.sqrt((origHeight * origWidth) / maxPixels);
  hBar = Math.max(factor, Math.floor(origHeight / beta / factor) * factor);
  wBar = Math.max(factor, Math.floor(origWidth / beta / factor) * factor);
}

// 按 merge_size 分组提取 patches
for (let gh = 0; gh < numPatchesH / patchMergeSize; gh++) {
  for (let gw = 0; gw < numPatchesW / patchMergeSize; gw++) {
    for (let mh = 0; mh < patchMergeSize; mh++) {
      for (let mw = 0; mw < patchMergeSize; mw++) {
        const normalized = (pixelValue - imageMean[c]) / imageStd[c];
      }
    }
  }
}
```

### 2. Token 序列构建

分段 tokenize 后拼接图像 token：

```javascript
const numImageTokens = Math.floor((gridT * gridH * gridW) / (mergeSize * mergeSize));

const part1 = "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<|vision_start|>";
const part2 = "<|vision_end|>文本<|im_end|>\n<|im_start|>assistant\n";

const ids1 = tokenizer(part1).input_ids;
const ids2 = tokenizer(part2).input_ids;
const imageTokens = new Array(numImageTokens).fill(imageTokenId);

const inputIds = [...ids1, ...imageTokens, ...ids2];
const mmTokenTypeIds = inputIds.map(id => id === imageTokenId ? 1 : 0);
```

### 3. RoPE 位置编码

完全移植 Python 的 `get_rope_index` 和 `get_vision_position_ids` 函数：

```javascript
function getVisionPositionIds(startPosition, gridThw, tempMergeSize, spatialMergeSize, timeInterval) {
  const llmGridT = Math.floor(gridThw[0] / tempMergeSize);
  const llmGridH = Math.floor(gridThw[1] / spatialMergeSize);
  const llmGridW = Math.floor(gridThw[2] / spatialMergeSize);
  
  const imageSeqLength = llmGridH * llmGridW * llmGridT;
  const positionWidth = [];
  const positionHeight = [];
  const positionTemporal = new Array(imageSeqLength).fill(startPosition * timeInterval);
  
  for (let t = 0; t < llmGridT; t++) {
    for (let h = 0; h < llmGridH; h++) {
      for (let w = 0; w < llmGridW; w++) {
        positionWidth.push(startPosition + w);
        positionHeight.push(startPosition + h);
      }
    }
  }
  
  return [positionTemporal, positionHeight, positionWidth];
}
```

### 4. 性能优化

直接在 TypedArray 上操作，避免大数组转换：

```javascript
// ❌ 错误：会导致栈溢出
const nextToken = lastLogits.indexOf(Math.max(...lastLogits));

// ✅ 正确：直接遍历 TypedArray
let maxVal = -Infinity;
let nextToken = 0;
const lastLogitsStart = (seqLen - 1) * vocabSize;
for (let i = 0; i < vocabSize; i++) {
  const val = logits.data[lastLogitsStart + i];
  if (val > maxVal) {
    maxVal = val;
    nextToken = i;
  }
}
```

## 性能对比

测试环境：macOS M 系列芯片，CPU 推理

| 指标 | Python | JavaScript |
|------|--------|------------|
| 首次图像处理 | ~30-60s | ~30-60s |
| 生成速度 | ~1.0 tok/s | ~1.1 tok/s |
| 输出质量 | 完全一致 | 完全一致 |
| 内存占用 | 较高 | 较低 |

## 技术难点

1. **smart_resize 算法**：动态计算目标尺寸，需要理解 min_pixels/max_pixels 逻辑
2. **patch 提取顺序**：按 merge_size 分组的嵌套循环，顺序错误会导致特征错位
3. **RoPE 位置编码**：3D 位置编码（temporal, height, width），需要精确计算
4. **Token 序列构建**：正确插入图像 token 并标记 mm_token_type_ids
5. **KV Cache 管理**：每步更新 past_key_values，维护生成状态

## 关键收获

- ✅ 图像预处理虽然复杂，但可以完全用 JavaScript 实现
- ✅ TypedArray 性能优秀，大数组操作直接在其上进行
- ✅ 通过 Python 输出对比，快速定位问题
- ✅ 核心算法完全移植，保证输出一致
- ⚠️ merge_size 分组顺序必须严格一致
- ⚠️ CHW 格式转换不能出错
- ⚠️ BigInt64Array 用于 int64 类型
- ⚠️ 边界检查避免数组越界

## 依赖

```json
{
  "@huggingface/transformers": "^3.3.1",
  "onnxruntime-node": "^1.20.1",
  "sharp": "^0.34.5"
}
```

## Python 版本对比

Python 版本见 `run.py`，使用方法：

```bash
conda activate qwenvl3b
python run.py
```
