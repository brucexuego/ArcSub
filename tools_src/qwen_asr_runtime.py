from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

_QWEN_BYTE_DECODER: Optional[Dict[str, int]] = None


def normalize_qwen_language(raw: Optional[str], supported_languages: Dict[str, str]) -> Optional[str]:
    value = str(raw or "").strip()
    if not value:
        return None

    lower = value.lower().replace("_", "-")
    if lower in ("auto", "none", "null"):
        return None

    alias_map = {
        "ar": "Arabic",
        "arabic": "Arabic",
        "cs": "Czech",
        "czech": "Czech",
        "da": "Danish",
        "danish": "Danish",
        "de": "German",
        "german": "German",
        "el": "Greek",
        "greek": "Greek",
        "en": "English",
        "english": "English",
        "es": "Spanish",
        "spanish": "Spanish",
        "fa": "Persian",
        "persian": "Persian",
        "fi": "Finnish",
        "finnish": "Finnish",
        "fil": "Filipino",
        "filipino": "Filipino",
        "fr": "French",
        "french": "French",
        "hi": "Hindi",
        "hindi": "Hindi",
        "hu": "Hungarian",
        "hungarian": "Hungarian",
        "id": "Indonesian",
        "indonesian": "Indonesian",
        "it": "Italian",
        "italian": "Italian",
        "ja": "Japanese",
        "japanese": "Japanese",
        "jp": "Japanese",
        "ko": "Korean",
        "korean": "Korean",
        "kr": "Korean",
        "mk": "Macedonian",
        "macedonian": "Macedonian",
        "ms": "Malay",
        "malay": "Malay",
        "nl": "Dutch",
        "dutch": "Dutch",
        "pl": "Polish",
        "polish": "Polish",
        "pt": "Portuguese",
        "portuguese": "Portuguese",
        "ro": "Romanian",
        "romanian": "Romanian",
        "ru": "Russian",
        "russian": "Russian",
        "sv": "Swedish",
        "swedish": "Swedish",
        "th": "Thai",
        "thai": "Thai",
        "tr": "Turkish",
        "turkish": "Turkish",
        "vi": "Vietnamese",
        "vietnamese": "Vietnamese",
        "yue": "Cantonese",
        "cantonese": "Cantonese",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh-hans": "Chinese",
        "zh-hant": "Chinese",
        "zh-hk": "Cantonese",
        "zh-mo": "Cantonese",
        "zh-sg": "Chinese",
        "zh-tw": "Chinese",
        "chinese": "Chinese",
    }

    canonical = supported_languages.get(lower) or supported_languages.get(value) or alias_map.get(lower)
    if not canonical:
        supported = ", ".join(sorted(set(supported_languages.values())))
        raise RuntimeError(f'Qwen3-ASR does not support language "{value}". Supported languages: {supported}')
    return canonical


def _get_qwen_byte_decoder_map() -> Dict[str, int]:
    global _QWEN_BYTE_DECODER
    if _QWEN_BYTE_DECODER is not None:
        return _QWEN_BYTE_DECODER

    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    char_values = list(byte_values)
    extra = 0
    for value in range(256):
        if value in byte_values:
            continue
        byte_values.append(value)
        char_values.append(256 + extra)
        extra += 1

    _QWEN_BYTE_DECODER = {chr(code_point): byte_values[index] for index, code_point in enumerate(char_values)}
    return _QWEN_BYTE_DECODER


@dataclass
class QwenAsrRuntime:
    np: Any
    feature_extractor: Any
    audio_encoder: Any
    thinker_embeddings: Any
    decoder_prefill: Any
    decoder_kv: Any
    prompt_template: Dict[str, Any]
    token_id_to_content: Dict[int, str]
    special_token_ids: set[int]
    eos_token_ids: set[int]
    supported_languages: Dict[str, str]
    vocab_size: int

    @classmethod
    def load(cls, model_dir: Any, ov: Any, np: Any, whisper_feature_extractor_cls: Any) -> "QwenAsrRuntime":
        import json

        prompt_template = json.loads((model_dir / "prompt_template.json").read_text(encoding="utf-8"))
        preprocessor_config = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
        tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        generation_config_path = model_dir / "generation_config.json"
        generation_config = (
            json.loads(generation_config_path.read_text(encoding="utf-8"))
            if generation_config_path.exists()
            else {}
        )

        preprocessor_config.pop("processor_class", None)
        preprocessor_config.setdefault("sampling_rate", 16000)
        feature_extractor = whisper_feature_extractor_cls(**preprocessor_config)

        token_id_to_content: Dict[int, str] = {int(token_id): token for token, token_id in vocab.items()}
        special_token_ids = set(int(value) for value in prompt_template.get("special_ids", []))
        for token_id, item in (tokenizer_config.get("added_tokens_decoder") or {}).items():
            numeric_id = int(token_id)
            content = str(item.get("content") or "")
            if content:
                token_id_to_content[numeric_id] = content
            if item.get("special"):
                special_token_ids.add(numeric_id)

        asr_text_id = prompt_template.get("asr_text_id")
        if isinstance(asr_text_id, int):
            token_id_to_content[asr_text_id] = "<asr_text>"

        eos_token_ids: set[int] = set()
        configured_eos = generation_config.get("eos_token_id")
        if isinstance(configured_eos, list):
            eos_token_ids.update(int(item) for item in configured_eos)
        elif isinstance(configured_eos, int):
            eos_token_ids.add(configured_eos)
        for key in ("eos_id", "eot_id"):
            value = prompt_template.get(key)
            if isinstance(value, int):
                eos_token_ids.add(value)

        supported_languages: Dict[str, str] = {}
        for item in prompt_template.get("supported_languages", []):
            normalized = str(item or "").strip()
            if not normalized:
                continue
            supported_languages[normalized] = normalized
            supported_languages[normalized.lower()] = normalized

        core = ov.Core()

        def compile_model(file_name: str):
            return core.compile_model(str(model_dir / file_name), "AUTO")

        vocab_size = (max(token_id_to_content.keys()) if token_id_to_content else 151935) + 1
        return cls(
            np=np,
            feature_extractor=feature_extractor,
            audio_encoder=compile_model("audio_encoder_model.xml"),
            thinker_embeddings=compile_model("thinker_embeddings_model.xml"),
            decoder_prefill=compile_model("decoder_prefill_kv_model.xml"),
            decoder_kv=compile_model("decoder_kv_model.xml"),
            prompt_template=prompt_template,
            token_id_to_content=token_id_to_content,
            special_token_ids=special_token_ids,
            eos_token_ids=eos_token_ids,
            supported_languages=supported_languages,
            vocab_size=vocab_size,
        )

    @staticmethod
    def _infer_outputs(compiled_model: Any, inputs: list[Any]) -> list[Any]:
        return list(compiled_model(inputs).values())

    def _embed_token_ids(self, token_ids: list[int]):
        if not token_ids:
            return self.np.zeros((1, 0, 2048), dtype=self.np.float32)
        input_ids = self.np.asarray([token_ids], dtype=self.np.int64)
        return self.np.asarray(self._infer_outputs(self.thinker_embeddings, [input_ids])[0], dtype=self.np.float32)

    def _encode_audio(self, audio: Any):
        max_length = int(self.prompt_template.get("n_samples") or 160000)
        features = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="np",
            truncation=True,
            max_length=max_length,
        )
        input_features = self.np.asarray(features.input_features, dtype=self.np.float32)
        mel = input_features[0]
        encoded = self.np.asarray(self._infer_outputs(self.audio_encoder, [mel])[0], dtype=self.np.float32)
        if encoded.ndim != 3 or encoded.shape[0] != 1:
            raise RuntimeError("Qwen3-ASR audio encoder returned invalid embeddings.")
        return encoded

    def _pick_greedy_token(self, logits: Any) -> int:
        token_logits = logits[0, -1, :] if logits.ndim == 3 else logits.reshape(-1)
        if token_logits.shape[-1] > self.vocab_size:
            token_logits = token_logits[-self.vocab_size :]
        return int(token_logits.argmax())

    def _decode_token_string(self, token: str) -> str:
        if not token:
            return ""
        byte_decoder = _get_qwen_byte_decoder_map()
        byte_values = [byte_decoder[symbol] for symbol in token if symbol in byte_decoder]
        return bytes(byte_values).decode("utf-8", errors="ignore") if byte_values else ""

    def _decode_token_ids(self, token_ids: list[int], keep_special: bool = False) -> str:
        parts: list[str] = []
        for token_id in token_ids:
            content = self.token_id_to_content.get(token_id)
            if not content:
                continue
            if token_id in self.special_token_ids:
                if keep_special:
                    parts.append(content)
                continue
            parts.append(self._decode_token_string(content))
        return "".join(parts)

    def _parse_transcript(self, generated_token_ids: list[int]) -> Dict[str, Any]:
        marker_id = self.prompt_template.get("asr_text_id")
        marker_index = generated_token_ids.index(marker_id) if marker_id in generated_token_ids else -1
        prefix_ids = generated_token_ids[:marker_index] if marker_index >= 0 else []
        transcript_ids = generated_token_ids[marker_index + 1 :] if marker_index >= 0 else generated_token_ids
        return {
            "text": self._decode_token_ids(transcript_ids).strip(),
            "rawPrefix": self._decode_token_ids(prefix_ids, keep_special=True).strip(),
        }

    def transcribe_chunk(self, audio_chunk: Any, language: Optional[str]) -> Dict[str, Any]:
        normalized_language = normalize_qwen_language(language, self.supported_languages)
        prefix_ids = list(self.prompt_template.get("prefix_ids") or [])
        suffix_ids = list(self.prompt_template.get("suffix_ids") or [])
        if normalized_language:
            suffix_ids.extend(self.prompt_template.get("language_suffix_ids", {}).get(normalized_language, []))

        prefix_embeddings = self._embed_token_ids(prefix_ids)
        suffix_embeddings = self._embed_token_ids(suffix_ids)
        audio_embeddings = self._encode_audio(audio_chunk)
        prefill_embeddings = self.np.concatenate([prefix_embeddings, audio_embeddings, suffix_embeddings], axis=1)
        position_ids = self.np.arange(prefill_embeddings.shape[1], dtype=self.np.int64).reshape(1, -1)
        prefill_outputs = self._infer_outputs(self.decoder_prefill, [prefill_embeddings, position_ids])

        logits = self.np.asarray(prefill_outputs[0], dtype=self.np.float32)
        past_keys = self.np.asarray(prefill_outputs[1], dtype=self.np.float32)
        past_values = self.np.asarray(prefill_outputs[2], dtype=self.np.float32)

        generated_token_ids: list[int] = []
        current_position = int(prefill_embeddings.shape[1])
        max_new_tokens = max(16, min(int(os.environ.get("OPENVINO_QWEN3_ASR_MAX_NEW_TOKENS", "256") or "256"), 4096))
        consecutive_special_tokens = 0

        for _ in range(max_new_tokens):
            next_token_id = self._pick_greedy_token(logits)
            if next_token_id in self.eos_token_ids:
                break
            generated_token_ids.append(next_token_id)
            if next_token_id in self.special_token_ids:
                consecutive_special_tokens += 1
                if consecutive_special_tokens >= 24:
                    break
            else:
                consecutive_special_tokens = 0

            token_embeddings = self._embed_token_ids([next_token_id])
            position_tensor = self.np.asarray([[current_position]], dtype=self.np.int64)
            decode_outputs = self._infer_outputs(self.decoder_kv, [token_embeddings, position_tensor, past_keys, past_values])
            logits = self.np.asarray(decode_outputs[0], dtype=self.np.float32)
            past_keys = self.np.asarray(decode_outputs[1], dtype=self.np.float32)
            past_values = self.np.asarray(decode_outputs[2], dtype=self.np.float32)
            current_position += 1

        return self._parse_transcript(generated_token_ids)
