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
  
  // 计算调整后的尺寸
  const aspectRatio = origWidth / origHeight;
  let targetWidth, targetHeight;
  
  // 简化：基于 min/max pixels 计算目标尺寸
  const totalPixels = origWidth * origHeight;
  if (totalPixels < minPixels) {
    const scale = Math.sqrt(minPixels / totalPixels);
    targetWidth = Math.round(origWidth * scale);
    targetHeight = Math.round(origHeight * scale);
  } else if (totalPixels > maxPixels) {
    const scale = Math.sqrt(maxPixels / totalPixels);
    targetWidth = Math.round(origWidth * scale);
    targetHeight = Math.round(origHeight * scale);
  } else {
    targetWidth = origWidth;
    targetHeight = origHeight;
  }
  
  // 调整为 patch_size 的倍数
  targetWidth = Math.round(targetWidth / patchSize) * patchSize;
  targetHeight = Math.round(targetHeight / patchSize) * patchSize;
  
  // 计算网格大小
  const gridH = targetHeight / patchSize;
  const gridW = targetWidth / patchSize;
  const gridT = 1; // 单帧图像
  
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
  const image = await RawImage.read(imagePath);
  const processInfo = processImage(image, config);
  
  // Resize 图像
  const resized = await image.resize(processInfo.targetWidth, processInfo.targetHeight);
  
  // 转换为 CHW 格式并归一化
  const { width, height, channels } = resized;
  const data = resized.data;
  
  // 提取 patches
  const patchSize = processInfo.patchSize;
  const numPatchesH = processInfo.gridH;
  const numPatchesW = processInfo.gridW;
  const numPatches = numPatchesH * numPatchesW;
  
  // 每个 patch 的维度
  const patchDim = channels * processInfo.temporalPatchSize * patchSize * patchSize;
  const pixelValues = new Float32Array(numPatches * patchDim);
  
  let patchIdx = 0;
  for (let ph = 0; ph < numPatchesH; ph++) {
    for (let pw = 0; pw < numPatchesW; pw++) {
      const startY = ph * patchSize;
      const startX = pw * patchSize;
      
      let offset = 0;
      // 提取 patch (CHW 格式)
      for (let c = 0; c < channels; c++) {
        for (let y = 0; y < patchSize; y++) {
          for (let x = 0; x < patchSize; x++) {
            const pixelY = startY + y;
            const pixelX = startX + x;
            const pixelIdx = (pixelY * width + pixelX) * channels + c;
            
            // 归一化
            const pixelValue = data[pixelIdx] / 255.0;
            const normalized = (pixelValue - processInfo.imageMean[c]) / processInfo.imageStd[c];
            
            pixelValues[patchIdx * patchDim + offset] = normalized;
            offset++;
          }
        }
      }
      patchIdx++;
    }
  }
  
  return {
    pixelValues,
    imageGridThw: [processInfo.gridT, processInfo.gridH, processInfo.gridW],
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
// 直接使用 Python processor 计算出的尺寸
// 从 debug 输出我们知道：pixel_values shape: (5476, 1176), image_grid_thw: [[1, 74, 74]]
const imageData = {
  imageGridThw: [1, 74, 74],
  numPatches: 5476,
  patchDim: 1176
};

// 使用 RawImage 读取并处理图像
const image = await RawImage.read('./test.jpg');
// 计算目标尺寸：74 * 14 = 1036
const targetSize = 74 * 14; // 1036x1036
const resized = await image.resize(targetSize, targetSize);

// 提取 patches 并归一化
const { width, height, channels } = resized;
const data = resized.data;
const patchSize = 14;
const numPatchesH = 74;
const numPatchesW = 74;

const imageMean = [0.48145466, 0.4578275, 0.40821073];
const imageStd = [0.26862954, 0.26130258, 0.27577711];

const pixelValues = new Float32Array(imageData.numPatches * imageData.patchDim);

let patchIdx = 0;
for (let ph = 0; ph < numPatchesH; ph++) {
  for (let pw = 0; pw < numPatchesW; pw++) {
    const startY = ph * patchSize;
    const startX = pw * patchSize;
    
    let offset = 0;
    // CHW 格式
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < patchSize; y++) {
        for (let x = 0; x < patchSize; x++) {
          const pixelY = startY + y;
          const pixelX = startX + x;
          const pixelIdx = (pixelY * width + pixelX) * channels + c;
          
          const pixelValue = data[pixelIdx] / 255.0;
          const normalized = (pixelValue - imageMean[c]) / imageStd[c];
          
          pixelValues[patchIdx * imageData.patchDim + offset] = normalized;
          offset++;
        }
      }
    }
    patchIdx++;
  }
}

imageData.pixelValues = pixelValues;

console.log('准备文本输入...');
const messages = [
  {
    role: "user",
    content: [
      { type: "image", image: "./34599220_182808366107_2.jpg" },
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

// 替换 <|image_pad|> 为多个图像 token
// 直接构建 token ids 而不是文本
const textBeforeImage = tokenizer.apply_chat_template([{
  role: "user",
  content: [{ type: "text", text: "" }]
}], { add_generation_prompt: false, tokenize: false });

const textAfterImage = "描述这个图片内容，用中文回答。<|im_end|>\n<|im_start|>assistant\n";

const tokensBeforeImage = tokenizer(textBeforeImage.replace("<|im_end|>", "<|vision_start|>"), { return_tensors: 'pt' });
const tokensAfterImage = tokenizer("<|vision_end|>" + textAfterImage, { return_tensors: 'pt' });

const inputIdsBeforeImage = Array.from(tokensBeforeImage.input_ids.data).map(x => Number(x));
const inputIdsAfterImage = Array.from(tokensAfterImage.input_ids.data).map(x => Number(x));

// 构建完整的 input_ids：before + image_tokens + after
const imageTokens = new Array(numImageTokens).fill(imageTokenId);
const fullInputIds = [...inputIdsBeforeImage, ...imageTokens, ...inputIdsAfterImage];

const inputIds = fullInputIds;
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
