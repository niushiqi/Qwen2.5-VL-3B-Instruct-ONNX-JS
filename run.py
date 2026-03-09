import itertools
import os
import numpy as np
import onnxruntime

from modelscope import snapshot_download
from modelscope import AutoConfig, AutoProcessor, GenerationConfig


def get_vision_position_ids(start_position, grid_thw, temp_merge_size=1, spatial_merge_size=1, time_interval=1):
    llm_grid_t = grid_thw[0] // temp_merge_size
    llm_grid_h = grid_thw[1] // spatial_merge_size
    llm_grid_w = grid_thw[2] // spatial_merge_size

    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = np.tile(np.arange(start_position, start_position + llm_grid_w), llm_grid_h * llm_grid_t)
    position_height = np.repeat(np.arange(start_position, start_position + llm_grid_h), llm_grid_w * llm_grid_t)
    position_temporal = np.full(image_seq_length, start_position, dtype=np.int64) * time_interval
    return np.stack([position_temporal, position_height, position_width], axis=0)


def get_rope_index(input_ids, mm_token_type_ids, image_grid_thw=None, video_grid_thw=None,
                   second_per_grid_ts=None, attention_mask=None, spatial_merge_size=2, tokens_per_second=25):
    batch_size, seq_len = input_ids.shape
    position_ids = np.zeros((3, batch_size, seq_len), dtype=np.int64)
    mrope_position_deltas = []

    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }
    second_per_grid_ts_iter = iter(second_per_grid_ts) if second_per_grid_ts is not None else iter([1] * seq_len)

    for batch_idx in range(batch_size):
        current_input_ids = input_ids[batch_idx]
        input_token_type = mm_token_type_ids[batch_idx]

        if attention_mask is not None:
            mask = attention_mask[batch_idx].astype(bool)
            current_input_ids = current_input_ids[mask]
            input_token_type = input_token_type[mask]

        input_type_groups = []
        for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            group = list(group)
            input_type_groups.append((key, group[0][0], group[-1][0] + 1))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_groups:
            if modality_type == 0:  # text
                text_len = end_idx - start_idx
                text_pos = np.arange(current_pos, current_pos + text_len, dtype=np.int64)
                llm_pos_ids_list.append(np.tile(text_pos, (3, 1)))
                current_pos += text_len
            else:  # image (1) or video (2)
                grid_thw = next(grid_iters[modality_type])
                time_interval = tokens_per_second * int(next(second_per_grid_ts_iter))
                vision_pos = get_vision_position_ids(current_pos, grid_thw, 1, spatial_merge_size, time_interval)
                llm_pos_ids_list.append(vision_pos)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1)  # (3, total_tokens)
        if attention_mask is not None:
            position_ids[:, batch_idx, attention_mask[batch_idx].astype(bool)] = llm_positions
        else:
            position_ids[:, batch_idx] = llm_positions

        mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))

    mrope_position_deltas = np.array(mrope_position_deltas, dtype=np.int64).reshape(-1, 1)
    return position_ids, mrope_position_deltas


# 1. Load models
## Define Model ID
model_id = "onnx-community/Qwen2.5-VL-3B-Instruct-ONNX"

## Load config, processor, and generation config
config = AutoConfig.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
generation_config = GenerationConfig.from_pretrained(model_id)

## Select model precisions
vision_encoder_path = "vision_encoder_q4.onnx"
embed_tokens_path = "embed_tokens_q4.onnx"
decoder_model_path = "decoder_model_merged_q4.onnx"

## Download ONNX models
print("Downloading ONNX models...")
onnx_dir = snapshot_download(
    repo_id=model_id,
    allow_patterns=[
        # Download requested graphs and weights
        f"onnx/{vision_encoder_path}*",
        f"onnx/{embed_tokens_path}*",
        f"onnx/{decoder_model_path}*",
    ]
)

## Load sessions
vision_session = onnxruntime.InferenceSession(os.path.join(onnx_dir, "onnx", vision_encoder_path))
embed_session = onnxruntime.InferenceSession(os.path.join(onnx_dir, "onnx", embed_tokens_path))
decoder_session = onnxruntime.InferenceSession(os.path.join(onnx_dir, "onnx", decoder_model_path))

## Set config values
text_config = config.text_config
num_key_value_heads = text_config.num_key_value_heads
head_dim = text_config.hidden_size // text_config.num_attention_heads
image_token_id = config.image_token_id
eos_token_id = generation_config.eos_token_id
spatial_merge_size = config.vision_config.spatial_merge_size
tokens_per_second = config.vision_config.tokens_per_second

# 2. Prepare inputs
## Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            {"type": "text", "text": "Describe this image."}
        ]
    },
]

## Apply chat template
pt_inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.cpu().numpy() for k, v in pt_inputs.items()}

## Calculate vision positions and RoPE deltas
position_ids, rope_deltas = get_rope_index(
    input_ids=inputs["input_ids"],
    mm_token_type_ids=inputs.get("mm_token_type_ids"),
    image_grid_thw=inputs.get("image_grid_thw"),
    attention_mask=inputs["attention_mask"],
    spatial_merge_size=spatial_merge_size,
    tokens_per_second=tokens_per_second,
)

## Prepare decoder variables
batch_size = inputs['input_ids'].shape[0]
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

## Initialize past_key_values cache
past_key_values = {
    inp.name: np.zeros(
        [batch_size, num_key_value_heads, 0, head_dim],
        dtype=np.float16 if "float16" in inp.type else np.float32
    )
    for inp in decoder_session.get_inputs()
    if inp.name.startswith("past_key_values")
}

# 3. Generation loop
max_new_tokens = 512
generated_tokens = np.array([[]], dtype=np.int64)
image_features = None

print("Generating...")
for step in range(max_new_tokens):
    ## Generate text embeddings
    inputs_embeds = embed_session.run(None, {'input_ids': input_ids})[0]

    ## Compute and inject vision features (only on the very first step)
    if image_features is None and "pixel_values" in inputs:
        vision_inputs = {"pixel_values": inputs["pixel_values"]}
        vision_input_names = {inp.name for inp in vision_session.get_inputs()}

        for optional_input in ["pixel_attention_mask", "spatial_shapes", "image_sizes", "image_grid_thw"]:
            if optional_input in vision_input_names and optional_input in inputs:
                vision_inputs[optional_input] = inputs[optional_input]

        image_features = vision_session.run(None, vision_inputs)[0]

        # Merge vision embeddings into the text embedding sequence
        inputs_embeds[input_ids == image_token_id] = image_features.reshape(-1, image_features.shape[-1])

    ## Run decoder step
    outputs = decoder_session.run(None, dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        **past_key_values,
    ))
    logits, present_key_values = outputs[0], outputs[1:]

    ## Update states for the next iteration
    next_token = logits[:, -1].argmax(-1, keepdims=True)
    input_ids = next_token

    attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=attention_mask.dtype)], axis=-1)

    ## Re-calculate positional IDs and apply RoPE deltas for the generated token
    text_positions = np.cumsum(attention_mask, axis=-1, dtype=np.int64) - 1
    text_positions = np.clip(text_positions, 0, None)[:, -1:]
    position_ids = np.broadcast_to(text_positions[None, ...], (3,) + text_positions.shape) + rope_deltas[None, ...]

    for j, key in enumerate(past_key_values):
        past_key_values[key] = present_key_values[j]

    generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)

    ## (Optional) Streaming
    print(processor.decode(input_ids[0]), end='', flush=True)

    if np.isin(input_ids, eos_token_id).all():
        break
print()


# 4. Output result
print("\n--- Final Decoded Output ---")
print(processor.decode(generated_tokens[0], skip_special_tokens=True))