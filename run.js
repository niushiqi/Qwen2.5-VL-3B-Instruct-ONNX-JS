import * as ort from 'onnxruntime-node';
import { AutoTokenizer, RawImage } from '@huggingface/transformers';
import fs from 'fs/promises';
import path from 'path';

const MODEL_DIR = '/Users/mindtower/.cache/modelscope/hub/models/onnx-community/Qwen2___5-VL-3B-Instruct-ONNX';
const ONNX_DIR = path.join(MODEL_DIR, 'onnx');

const config = JSON.parse(await fs.readFile(path.join(MODEL_DIR, 'config.json'), 'utf-8'));
const generationConfig = JSON.parse(await fs.readFile(path.join(MODEL_DIR, 'generation_config.json'), 'utf-8'));
const preprocessorConfig = JSON.parse(await fs.readFile(path.join(MODEL_DIR, 'preprocessor_config.json'), 'utf-8'));

const textConfig = config.text_config;
const visionConfig = config.vision_config;
const numKeyValueHeads = textConfig.num_key_value_heads;
const headDim = Math.floor(textConfig.hidden_size / textConfig.num_attention_heads);
const numLayers = textConfig.num_hidden_layers;
const imageTokenId = config.image_token_id;
const eosTokenId = Array.isArray(generationConfig.eos_token_id) 
  ? generationConfig.eos_token_id 
  : [generationConfig.eos_token_id];
const spatialMergeSize = visionConfig.spatial_merge_size;
const tokensPerSecond = visionConfig.tokens_per_second;

// 图像预处理函数
function processImage(image, config) {
  const minPixels = config.min_pixels || 3136;
  const maxPixels = config.max_pixels || 12845056;
  const patchSize = config.patch_size || 14;
  const temporalPatchSize = config.temporal_patch_size || 2;
  const mergeSize = config.merge_size || 2;
  const imageMean = config.image_mean || [0.48145466, 0.4578275, 0.40821073];
  const imageStd = config.image_std || [0.26862954, 0.26130258, 0.27577711];

  const origWidth = image.width;
  const origHeight = image.height;

  // smart_resize 的 factor 是 patch_size * merge_size
  const factor = patchSize * mergeSize;

  // smart_resize 逻辑
  let hBar = Math.round(origHeight / factor) * factor;
  let wBar = Math.round(origWidth / factor) * factor;

  if (hBar * wBar > maxPixels) {
    const beta = Math.sqrt((origHeight * origWidth) / maxPixels);
    hBar = Math.max(factor, Math.floor(origHeight / beta / factor) * factor);
    wBar = Math.max(factor, Math.floor(origWidth / beta / factor) * factor);
  } else if (hBar * wBar < minPixels) {
    const beta = Math.sqrt(minPixels / (origHeight * origWidth));
    hBar = Math.ceil(origHeight * beta / factor) * factor;
    wBar = Math.ceil(origWidth * beta / factor) * factor;
  }

  const targetWidth = wBar;
  const targetHeight = hBar;

  // 计算网格大小
  const gridH = targetHeight / patchSize;
  const gridW = targetWidth / patchSize;
  const gridT = 1;

  return {
    targetWidth,
    targetHeight,
    gridT,
    gridH,
    gridW,
    imageMean,
    imageStd,
    patchSize,
    temporalPatchSize,
    mergeSize
  };
}

async function preprocessImage(imagePath, config) {
  // 1. 读取图像
  const image = await RawImage.read(imagePath);

  // 2. 参数
  const minPixels = config.min_pixels || 3136;
  const maxPixels = config.max_pixels || 12845056;
  const patchSize = config.patch_size || 14;
  const temporalPatchSize = config.temporal_patch_size || 2;
  const mergeSize = config.merge_size || 2;
  const imageMean = config.image_mean || [0.48145466, 0.4578275, 0.40821073];
  const imageStd = config.image_std || [0.26862954, 0.26130258, 0.27577711];
  const rescaleFactor = 1 / 255;

  const origHeight = image.height;
  const origWidth = image.width;

  // 3. smart_resize
  const factor = patchSize * mergeSize;

  // 检查宽高比
  if (Math.max(origHeight, origWidth) / Math.min(origHeight, origWidth) > 200) {
    throw new Error(`absolute aspect ratio must be smaller than 200, got ${Math.max(origHeight, origWidth) / Math.min(origHeight, origWidth)}`);
  }

  let hBar = Math.round(origHeight / factor) * factor;
  let wBar = Math.round(origWidth / factor) * factor;

  if (hBar * wBar > maxPixels) {
    const beta = Math.sqrt((origHeight * origWidth) / maxPixels);
    hBar = Math.max(factor, Math.floor(origHeight / beta / factor) * factor);
    wBar = Math.max(factor, Math.floor(origWidth / beta / factor) * factor);
  } else if (hBar * wBar < minPixels) {
    const beta = Math.sqrt(minPixels / (origHeight * origWidth));
    hBar = Math.ceil(origHeight * beta / factor) * factor;
    wBar = Math.ceil(origWidth * beta / factor) * factor;
  }

  const resizedHeight = hBar;
  const resizedWidth = wBar;

  // 4. Resize (bicubic)
  const resized = await image.resize(resizedWidth, resizedHeight, { resample: 3 });
  const { width, height, channels } = resized;
  const data = resized.data;  // HWC 格式

  // 5. Rescale + Normalize + 转 CHW (data_format = ChannelDimension.FIRST)
  const processedImage = new Float32Array(channels * height * width);

  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const hwcIdx = (y * width + x) * channels + c;
        const chwIdx = c * height * width + y * width + x;

        // Rescale
        const rescaled = data[hwcIdx] * rescaleFactor;

        // Normalize
        const normalized = (rescaled - imageMean[c]) / imageStd[c];

        processedImage[chwIdx] = normalized;
      }
    }
  }

  // 6. patches = np.array([processedImage])
  // 对于单张图片，patches shape: (1, C, H, W)
  let numImages = 1;
  let patches = new Float32Array(numImages * channels * height * width);
  patches.set(processedImage, 0);

  // 7. 填充到 temporal_patch_size 的倍数
  if (numImages % temporalPatchSize !== 0) {
    const repeatsNeeded = temporalPatchSize - (numImages % temporalPatchSize);
    const newPatches = new Float32Array((numImages + repeatsNeeded) * channels * height * width);
    newPatches.set(patches, 0);

    // 重复最后一张图
    for (let i = 0; i < repeatsNeeded; i++) {
      newPatches.set(processedImage, (numImages + i) * channels * height * width);
    }

    patches = newPatches;
    numImages += repeatsNeeded;
  }

  // 8. 计算 grid
  const gridT = Math.floor(numImages / temporalPatchSize);
  const gridH = Math.floor(resizedHeight / patchSize);
  const gridW = Math.floor(resizedWidth / patchSize);
  const channel = channels;

  // 9. Reshape + Transpose + Flatten
  // Reshape: (numImages, C, H, W) -> (grid_t, temporal_patch_size, channel, grid_h//merge_size, merge_size, patch_size, grid_w//merge_size, merge_size, patch_size)
  // Transpose: (0, 3, 6, 4, 7, 2, 1, 5, 8)
  // Flatten: (grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)

  const numPatches = gridT * gridH * gridW;
  const patchDim = channel * temporalPatchSize * patchSize * patchSize;
  const flattenPatches = new Float32Array(numPatches * patchDim);

  // 直接按照 transpose 后的顺序遍历并填充
  let outputIdx = 0;

  for (let t = 0; t < gridT; t++) {
    for (let gh = 0; gh < gridH / mergeSize; gh++) {
      for (let gw = 0; gw < gridW / mergeSize; gw++) {
        for (let mh = 0; mh < mergeSize; mh++) {
          for (let mw = 0; mw < mergeSize; mw++) {
            for (let c = 0; c < channel; c++) {
              for (let tp = 0; tp < temporalPatchSize; tp++) {
                for (let ph = 0; ph < patchSize; ph++) {
                  for (let pw = 0; pw < patchSize; pw++) {
                    // 计算在原始 patches 中的位置
                    const imageIdx = t * temporalPatchSize + tp;
                    const y = (gh * mergeSize + mh) * patchSize + ph;
                    const x = (gw * mergeSize + mw) * patchSize + pw;
                    const srcIdx = imageIdx * channel * height * width + c * height * width + y * width + x;

                    flattenPatches[outputIdx++] = patches[srcIdx];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return {
    pixelValues: flattenPatches,
    imageGridThw: [gridT, gridH, gridW],
    numPatches,
    patchDim
  };
}

function getVisionPositionIds(startPosition, gridThw, tempMergeSize = 1, spatialMergeSize = 1, timeInterval = 1) {
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

function getRopeIndex(inputIds, mmTokenTypeIds, imageGridThw, videoGridThw, secondPerGridTs, attentionMask, spatialMergeSize, tokensPerSecond) {
  const [batchSize, seqLen] = [inputIds.length, inputIds[0].length];
  const positionIds = Array(3).fill(0).map(() => 
    Array(batchSize).fill(0).map(() => new Array(seqLen).fill(0))
  );
  const mropePositionDeltas = [];
  
  const gridIters = {
    1: imageGridThw ? [...imageGridThw] : [],
    2: videoGridThw ? [...videoGridThw] : []
  };
  const secondPerGridTsIter = secondPerGridTs || [];
  let secondIdx = 0;
  
  for (let batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    let currentInputIds = inputIds[batchIdx];
    let inputTokenType = mmTokenTypeIds[batchIdx];
    
    if (attentionMask) {
      const mask = attentionMask[batchIdx];
      currentInputIds = currentInputIds.filter((_, i) => mask[i] === 1);
      inputTokenType = inputTokenType.filter((_, i) => mask[i] === 1);
    }
    
    const inputTypeGroups = [];
    let i = 0;
    while (i < inputTokenType.length) {
      const modalityType = inputTokenType[i];
      const startIdx = i;
      while (i < inputTokenType.length && inputTokenType[i] === modalityType) {
        i++;
      }
      inputTypeGroups.push([modalityType, startIdx, i]);
    }
    
    let currentPos = 0;
    const llmPosIdsList = [];
    
    for (const [modalityType, startIdx, endIdx] of inputTypeGroups) {
      if (modalityType === 0) {
        const textLen = endIdx - startIdx;
        const textPos = Array.from({ length: textLen }, (_, i) => currentPos + i);
        llmPosIdsList.push([textPos, textPos, textPos]);
        currentPos += textLen;
      } else {
        const gridThw = gridIters[modalityType].shift();
        const timeInterval = tokensPerSecond * (secondPerGridTsIter[secondIdx++] || 1);
        const visionPos = getVisionPositionIds(currentPos, gridThw, 1, spatialMergeSize, timeInterval);
        llmPosIdsList.push(visionPos);
        currentPos += Math.floor(Math.max(gridThw[1], gridThw[2]) / spatialMergeSize);
      }
    }
    
    const llmPositions = [[], [], []];
    for (const posList of llmPosIdsList) {
      for (let dim = 0; dim < 3; dim++) {
        llmPositions[dim].push(...posList[dim]);
      }
    }
    
    if (attentionMask) {
      const mask = attentionMask[batchIdx];
      let posIdx = 0;
      for (let seqIdx = 0; seqIdx < seqLen; seqIdx++) {
        if (mask[seqIdx] === 1) {
          for (let dim = 0; dim < 3; dim++) {
            positionIds[dim][batchIdx][seqIdx] = llmPositions[dim][posIdx];
          }
          posIdx++;
        }
      }
    } else {
      for (let dim = 0; dim < 3; dim++) {
        positionIds[dim][batchIdx] = llmPositions[dim];
      }
    }
    
    mropePositionDeltas.push(Math.max(...llmPositions.flat()) + 1 - currentInputIds.length);
  }
  
  return [positionIds, mropePositionDeltas];
}

console.log('加载 ONNX 模型...');
const visionSession = await ort.InferenceSession.create(
  path.join(ONNX_DIR, 'vision_encoder_q4.onnx'),
  { executionProviders: ['cpu'] }
);
const embedSession = await ort.InferenceSession.create(
  path.join(ONNX_DIR, 'embed_tokens_q4.onnx'),
  { executionProviders: ['cpu'] }
);
const decoderSession = await ort.InferenceSession.create(
  path.join(ONNX_DIR, 'decoder_model_merged_q4.onnx'),
  { executionProviders: ['cpu'] }
);

console.log('加载 tokenizer...');
const tokenizer = await AutoTokenizer.from_pretrained(MODEL_DIR, {
  local_files_only: true
});

console.log('预处理图像...');
const imagePath = './test.jpg';
const image = await RawImage.read(imagePath);
const processInfo = processImage(image, preprocessorConfig);

// Resize 图像
const resized = await image.resize(processInfo.targetWidth, processInfo.targetHeight);

// 提取 patches 并归一化
const { width, height, channels } = resized;
const data = resized.data;
const patchSize = processInfo.patchSize;
const temporalPatchSize = processInfo.temporalPatchSize;
const numPatchesH = processInfo.gridH;
const numPatchesW = processInfo.gridW;
const numPatches = numPatchesH * numPatchesW;

// 每个 patch 的维度：channels * temporalPatchSize * patchSize * patchSize
const patchDim = channels * temporalPatchSize * patchSize * patchSize;
const pixelValues = new Float32Array(numPatches * patchDim);

console.log(`图像尺寸: ${width}x${height}, 网格: ${numPatchesH}x${numPatchesW}, patchDim: ${patchDim}`);

let patchIdx = 0;
const patchMergeSize = processInfo.mergeSize;

// 按照 merge_size 分组的顺序提取 patch
for (let gh = 0; gh < numPatchesH / patchMergeSize; gh++) {
  for (let gw = 0; gw < numPatchesW / patchMergeSize; gw++) {
    for (let mh = 0; mh < patchMergeSize; mh++) {
      for (let mw = 0; mw < patchMergeSize; mw++) {
        const ph = gh * patchMergeSize + mh;
        const pw = gw * patchMergeSize + mw;
        const startY = ph * patchSize;
        const startX = pw * patchSize;
        
        let offset = 0;
        // CHW 格式，加上时间维度重复
        for (let c = 0; c < channels; c++) {
          for (let t = 0; t < temporalPatchSize; t++) {
            for (let y = 0; y < patchSize; y++) {
              for (let x = 0; x < patchSize; x++) {
                const pixelY = startY + y;
                const pixelX = startX + x;
                const pixelIdx = (pixelY * width + pixelX) * channels + c;
                
                const pixelValue = data[pixelIdx] / 255.0;
                const normalized = (pixelValue - processInfo.imageMean[c]) / processInfo.imageStd[c];
                
                pixelValues[patchIdx * patchDim + offset] = normalized;
                offset++;
              }
            }
          }
        }
        patchIdx++;
      }
    }
  }
}

const imageData = {
  pixelValues,
  imageGridThw: [processInfo.gridT, processInfo.gridH, processInfo.gridW],
  numPatches,
  patchDim
};

console.log('准备文本输入...');
const messages = [
  {
    role: "user",
    content: [
      { type: "image", image: "./test.jpg" },
      { type: "text", text: "描述这个图片内容，用中文回答。" }
    ]
  }
];

let text = tokenizer.apply_chat_template(messages, { 
  add_generation_prompt: true,
  tokenize: false
});

// Qwen2VL 特殊处理：替换 <|image_pad|> 为实际的图像 token 数量
const mergeSize = 2; // spatial_merge_size
const gridT = imageData.imageGridThw[0];
const gridH = imageData.imageGridThw[1];
const gridW = imageData.imageGridThw[2];
const numImageTokens = Math.floor((gridT * gridH * gridW) / (mergeSize * mergeSize));

console.log(`图像 tokens 数量: ${numImageTokens}`);

// 构建正确的 input_ids
// 格式：<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<|vision_start|><image_tokens><|vision_end|>文本<|im_end|>\n<|im_start|>assistant\n
const part1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>";
const part2 = "<|vision_end|>描述这个图片内容，用中文回答。<|im_end|>\n<|im_start|>assistant\n";

const tokens1 = tokenizer(part1, { return_tensors: 'pt' });
const tokens2 = tokenizer(part2, { return_tensors: 'pt' });

const ids1 = Array.from(tokens1.input_ids.data).map(x => Number(x));
const ids2 = Array.from(tokens2.input_ids.data).map(x => Number(x));

// 构建完整的 input_ids：part1 + image_tokens + part2
const imageTokens = new Array(numImageTokens).fill(imageTokenId);
const inputIds = [...ids1, ...imageTokens, ...ids2];
const attentionMask = new Array(inputIds.length).fill(1);

console.log(`总 token 数: ${inputIds.length}, 其中图像 tokens: ${numImageTokens}`);

// 创建 mm_token_type_ids
const mmTokenTypeIds = inputIds.map(id => id === imageTokenId ? 1 : 0);

const batchSize = 1;
const inputIdsArray = [inputIds];
const attentionMaskArray = [attentionMask];
const mmTokenTypeIdsArray = [mmTokenTypeIds];
const imageGridThwArray = [imageData.imageGridThw];

const [positionIds, ropeDeltas] = getRopeIndex(
  inputIdsArray,
  mmTokenTypeIdsArray,
  imageGridThwArray,
  null,
  null,
  attentionMaskArray,
  spatialMergeSize,
  tokensPerSecond
);

const pastKeyValues = {};
for (let i = 0; i < numLayers; i++) {
  pastKeyValues[`past_key_values.${i}.key`] = new ort.Tensor(
    'float32',
    new Float32Array(batchSize * numKeyValueHeads * 0 * headDim),
    [batchSize, numKeyValueHeads, 0, headDim]
  );
  pastKeyValues[`past_key_values.${i}.value`] = new ort.Tensor(
    'float32',
    new Float32Array(batchSize * numKeyValueHeads * 0 * headDim),
    [batchSize, numKeyValueHeads, 0, headDim]
  );
}

console.log('开始生成...');
const maxNewTokens = 512;
const generatedTokens = [];
let imageFeatures = null;
let currentInputIds = inputIds;
let currentAttentionMask = attentionMask;
let currentPositionIds = positionIds;

const startTime = Date.now();
const tokenTimes = [];

for (let step = 0; step < maxNewTokens; step++) {
  const stepStart = Date.now();
  
  const inputIdsTensor = new ort.Tensor(
    'int64',
    new BigInt64Array(currentInputIds.map(x => BigInt(x))),
    [1, currentInputIds.length]
  );
  
  const embedOutputs = await embedSession.run({ input_ids: inputIdsTensor });
  let inputsEmbeds = embedOutputs.inputs_embeds;
  
  if (imageFeatures === null && step === 0) {
    console.log('处理图像（首次运行可能需要 30-60 秒）...');
    const visionStart = Date.now();
    
    const pixelValuesTensor = new ort.Tensor(
      'float32',
      imageData.pixelValues,
      [imageData.numPatches, imageData.patchDim]
    );
    
    const gridThwTensor = new ort.Tensor(
      'int64',
      new BigInt64Array(imageData.imageGridThw.map(x => BigInt(x))),
      [1, 3]
    );
    
    console.log('运行视觉编码器...');
    const visionOutputs = await visionSession.run({
      pixel_values: pixelValuesTensor,
      image_grid_thw: gridThwTensor
    });
    
    imageFeatures = visionOutputs.image_features;
    console.log(`✓ 图像处理完成，耗时 ${((Date.now() - visionStart) / 1000).toFixed(2)}s`);
    
    const embedsData = new Float32Array(inputsEmbeds.data);
    const featuresData = new Float32Array(imageFeatures.data);
    const embedDim = inputsEmbeds.dims[2];
    const numImageTokens = imageFeatures.dims[0];
    
    let featureIdx = 0;
    for (let i = 0; i < currentInputIds.length && featureIdx < numImageTokens; i++) {
      if (currentInputIds[i] === imageTokenId) {
        for (let d = 0; d < embedDim; d++) {
          embedsData[i * embedDim + d] = featuresData[featureIdx * embedDim + d];
        }
        featureIdx++;
      }
    }
    
    inputsEmbeds = new ort.Tensor('float32', embedsData, inputsEmbeds.dims);
  }
  
  const positionIdsTensor = new ort.Tensor(
    'int64',
    new BigInt64Array(currentPositionIds.flat(2).map(x => BigInt(x))),
    [3, 1, currentInputIds.length]
  );
  
  const attentionMaskTensor = new ort.Tensor(
    'int64',
    new BigInt64Array(currentAttentionMask.map(x => BigInt(x))),
    [1, currentAttentionMask.length]
  );
  
  const decoderInputs = {
    inputs_embeds: inputsEmbeds,
    attention_mask: attentionMaskTensor,
    position_ids: positionIdsTensor,
    ...pastKeyValues
  };
  
  const outputs = await decoderSession.run(decoderInputs);
  const logits = outputs.logits;
  
  const vocabSize = logits.dims[2];
  const seqLen = logits.dims[1];
  const lastLogitsStart = (seqLen - 1) * vocabSize;
  
  let maxVal = -Infinity;
  let nextToken = 0;
  for (let i = 0; i < vocabSize; i++) {
    const val = logits.data[lastLogitsStart + i];
    if (val > maxVal) {
      maxVal = val;
      nextToken = i;
    }
  }
  
  generatedTokens.push(nextToken);
  
  currentInputIds = [nextToken];
  currentAttentionMask.push(1);
  
  for (let i = 0; i < numLayers; i++) {
    pastKeyValues[`past_key_values.${i}.key`] = outputs[`present.${i}.key`];
    pastKeyValues[`past_key_values.${i}.value`] = outputs[`present.${i}.value`];
  }
  
  const textPositions = currentAttentionMask.map((_, i) => i);
  const lastPos = textPositions[textPositions.length - 1];
  currentPositionIds = [
    [[lastPos + ropeDeltas[0]]],
    [[lastPos + ropeDeltas[0]]],
    [[lastPos + ropeDeltas[0]]]
  ];
  
  const stepTime = Date.now() - stepStart;
  tokenTimes.push(stepTime);
  
  if (step > 0) {
    const avgTime = tokenTimes.slice(1).reduce((a, b) => a + b, 0) / (tokenTimes.length - 1);
    const tokensPerSec = 1000 / avgTime;
    process.stdout.write(`\rToken ${step + 1} | ${tokensPerSec.toFixed(2)} tok/s | ${(stepTime / 1000).toFixed(3)}s `);
  }
  
  const decoded = tokenizer.decode([nextToken], { skip_special_tokens: false });
  if (decoded.trim()) {
    process.stdout.write(decoded);
  }
  
  if (eosTokenId.includes(nextToken)) {
    break;
  }
}

const totalTime = (Date.now() - startTime) / 1000;
const avgSpeed = generatedTokens.length / totalTime;

console.log(`\n\n${'='.repeat(60)}`);
console.log('生成统计:');
console.log(`${'='.repeat(60)}`);
console.log(`生成 token 数: ${generatedTokens.length}`);
console.log(`总耗时: ${totalTime.toFixed(2)}s`);
console.log(`平均速度: ${avgSpeed.toFixed(2)} tok/s`);
console.log(`${'='.repeat(60)}`);

console.log('\n--- 最终解码输出 ---');
const finalOutput = tokenizer.decode(generatedTokens, { skip_special_tokens: true });
console.log(finalOutput);
console.log(`\n--- 输出长度: ${finalOutput.length} 字符 ---`);
