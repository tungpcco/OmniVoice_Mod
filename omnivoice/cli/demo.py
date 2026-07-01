#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
import os
import re
import tempfile
from typing import Any, Dict

# Configure Gradio temporary directory to be inside the workspace root.
# This resolves browser auto-download bugs caused by special characters, spaces,
# or short-name aliases (like NGUYEN~1) in standard system temporary directory paths.
_workspace_tmp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tmp"))
os.makedirs(_workspace_tmp, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _workspace_tmp

import gradio as gr
import numpy as np
import torch
import pydub
from pydub.effects import speedup

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    #parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument("--ip", default="localhost", help="Server IP (default: localhost).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    return parser


# ---------------------------------------------------------------------------
# SRT Parsing and Synthesis Helpers
# ---------------------------------------------------------------------------


def wrap_text_42(text: str) -> str:
    if len(text) <= 42:
        return text
    
    if " " in text:
        words = text.split(" ")
        total_len = len(text)
        half_len = total_len / 2
        
        current_len = 0
        best_split_idx = 1
        min_diff = float("inf")
        
        for i in range(1, len(words)):
            left_part = " ".join(words[:i])
            diff = abs(len(left_part) - half_len)
            if diff < min_diff:
                min_diff = diff
                best_split_idx = i
                
        line1 = " ".join(words[:best_split_idx])
        line2 = " ".join(words[best_split_idx:])
        return f"{line1}\n{line2}"
    else:
        mid = len(text) // 2
        return f"{text[:mid]}\n{text[mid:]}"


def format_srt_time(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    if millis > 999:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_from_chunks(chunks: list) -> str:
    srt_lines = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        if not text:
            continue
        
        text = wrap_text_42(text)
        
        ts = chunk.get("timestamp", (0.0, 0.0))
        if ts is None:
            ts = (0.0, 0.0)
        start, end = ts
        if start is None:
            start = 0.0
        if end is None:
            end = start + 2.0
            
        start_str = format_srt_time(start)
        end_str = format_srt_time(end)
        
        srt_lines.append(f"{i + 1}")
        srt_lines.append(f"{start_str} --> {end_str}")
        srt_lines.append(text)
        srt_lines.append("")
        
    return "\n".join(srt_lines)


def parse_srt(srt_path_or_content: str):
    if os.path.exists(srt_path_or_content):
        with open(srt_path_or_content, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    else:
        content = srt_path_or_content

    # Normalize line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    blocks = []
    lines = content.split('\n')
    
    current_block = {}
    state = 0  # 0: looking for index, 1: looking for timestamp, 2: reading text
    text_lines = []
    
    time_pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})')
    
    def time_to_ms(t_str):
        parts = t_str.replace(',', ':').split(':')
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2])
        ms = int(parts[3])
        return ((h * 3600 + m * 60 + s) * 1000) + ms

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            if state == 2:
                current_block['text'] = " ".join(text_lines).strip()
                if current_block.get('text'):
                    blocks.append(current_block)
                current_block = {}
                text_lines = []
                state = 0
            continue
            
        if state == 0:
            if line_strip.isdigit():
                current_block['index'] = int(line_strip)
                state = 1
        elif state == 1:
            match = time_pattern.search(line_strip)
            if match:
                current_block['start_ms'] = time_to_ms(match.group(1))
                current_block['end_ms'] = time_to_ms(match.group(2))
                state = 2
            else:
                if line_strip.isdigit():
                    current_block['index'] = int(line_strip)
        elif state == 2:
            text_lines.append(line_strip)
                
    if state == 2 and current_block:
        current_block['text'] = " ".join(text_lines).strip()
        if current_block.get('text'):
            blocks.append(current_block)
            
    blocks.sort(key=lambda x: x['start_ms'])
    return blocks


def synthesize_srt(
    srt_file_path,
    model,
    sampling_rate,
    language="English",
    ref_audio=None,
    ref_text=None,
    instruct=None,
    num_step=32,
    guidance_scale=2.0,
    denoise=True,
    speed=1.0,
    preprocess_prompt=True,
    postprocess_output=True,
    stretch_duration=False,
    stretch_pause_ms=200,
    progress=gr.Progress()
):
    logs_list = ["--- Starting new SRT synthesis session ---"]
    def append_log(msg):
        logging.info(msg)
        logs_list.append(msg)
        
    if not srt_file_path:
        append_log("Error: No SRT file provided.")
        yield None, "Please upload an SRT file.", "\n".join(logs_list)
        return
        
    append_log("Parsing SRT file...")
    yield None, "Parsing SRT...", "\n".join(logs_list)
    try:
        blocks = parse_srt(srt_file_path)
    except Exception as e:
        append_log(f"Error parsing SRT: {e}")
        yield None, f"Error parsing SRT: {e}", "\n".join(logs_list)
        return
        
    if not blocks:
        append_log("Error: No valid subtitle blocks found in the SRT file.")
        yield None, "No valid subtitle blocks found.", "\n".join(logs_list)
        return
        
    append_log(f"Parsed {len(blocks)} subtitle blocks successfully.")
    progress(0, desc="Initializing Voice Clone/Style...")
    yield None, "Initializing Voice Clone/Style...", "\n".join(logs_list)
    
    # Pre-calculate voice clone prompt if ref_audio is provided to avoid repeating ASR/tokenization
    clone_prompt = None
    if ref_audio:
        append_log("Pre-calculating voice clone prompt...")
        try:
            clone_prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text or None,
                preprocess_prompt=preprocess_prompt
            )
            append_log("Voice clone prompt generated successfully.")
        except Exception as e:
            append_log(f"Error initializing reference voice clone: {e}")
            yield None, f"Error initializing voice clone: {e}", "\n".join(logs_list)
            return

    # Determine total audio length
    total_duration_ms = blocks[-1]['end_ms']
    final_audio = pydub.AudioSegment.silent(duration=total_duration_ms, frame_rate=sampling_rate)
    
    # Common generation options - force postprocess_output=False to do custom padding/fading
    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale),
        denoise=bool(denoise),
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=False,
    )
    
    lang = language if (language and language != "Auto") else None
    
    # Generate each block
    num_blocks = len(blocks)
    for i, block in enumerate(blocks):
        block_text_snippet = block['text'][:40] + "..." if len(block['text']) > 40 else block['text']
        append_log(f"Block {i+1}/{num_blocks} ({block['start_ms']}ms -> {block['end_ms']}ms): '{block_text_snippet}'")
        progress((i / num_blocks), desc=f"Synthesizing block {i+1}/{num_blocks}...")
        yield None, f"Synthesizing block {i+1}/{num_blocks}...", "\n".join(logs_list)
        
        block_dur_ms = block['end_ms'] - block['start_ms']
        if stretch_duration:
            target_dur_ms = max(100, int(block_dur_ms - stretch_pause_ms))
            duration_val = target_dur_ms / 1000.0
        else:
            target_dur_ms = block_dur_ms
            duration_val = (block_dur_ms / 1000.0) + 0.3
            
        kw = {
            "text": block['text'],
            "language": lang,
            "generation_config": gen_config,
            "duration": duration_val
        }
        
        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
            
        max_retries = 3
        segment = None
        
        for retry in range(max_retries):
            # 1. Randomize PyTorch seed on retry to ensure stochastic path selection
            if retry > 0:
                import torch
                torch.manual_seed(torch.initial_seed() + retry + 1)
                
            # 2. Try default/auto voice on the final attempt if cloning/instruct causes failures
            block_kw = kw.copy()
            if retry == max_retries - 1:
                # Remove cloning/instruct prompts to fallback to default voice
                append_log(f"  -> Retry {retry+1}/{max_retries}: Falling back to default voice.")
            else:
                if retry > 0:
                    append_log(f"  -> Retry {retry+1}/{max_retries}: Retrying with shifted seed...")
                else:
                    append_log(f"  -> Attempt {retry+1}/{max_retries}...")
                
                if clone_prompt:
                    block_kw["voice_clone_prompt"] = clone_prompt
                if instruct and instruct.strip():
                    block_kw["instruct"] = instruct.strip()

            try:
                # Generate audio for the single block
                audio_out = model.generate(**block_kw)
                waveform = (audio_out[0] * 32767).astype(np.int16)
                
                # Strip core library's 100ms padding first
                pad_samples = int(0.1 * sampling_rate)
                if len(waveform) > 2 * pad_samples:
                    waveform_stripped = waveform[pad_samples : -pad_samples]
                else:
                    waveform_stripped = waveform
                
                # Verify speech energy distribution across 100ms windows
                waveform_float = waveform_stripped.astype(np.float32)
                win_len = int(0.1 * sampling_rate)
                num_wins = len(waveform_float) // win_len
                words = block['text'].split()
                
                is_empty = True
                if num_wins > 0 and len(words) > 0:
                    max_val = np.max(np.abs(waveform_float))
                    # VAD Check 1: Check that the signal is not absolute silence (min peak threshold 1000.0)
                    if max_val > 1000.0:
                        # Normalize to 32767 peak to make verification completely volume-invariant
                        waveform_norm = waveform_float / max_val * 32767.0
                        win_rms = [
                            np.sqrt(np.mean(waveform_norm[w * win_len : (w + 1) * win_len] ** 2))
                            for w in range(num_wins)
                        ]
                        max_rms = max(win_rms)
                        min_rms = min(win_rms)
                        
                        # VAD Check 2: Constant hum/noise filter
                        if min_rms < max_rms * 0.35:
                            threshold = max_rms * 0.15
                            active_wins = sum(1 for r in win_rms if r > threshold)
                            
                            # VAD Check 3: Word count constraint
                            expected_active_wins = max(2, int(len(words) * 1.2))
                            if active_wins >= expected_active_wins:
                                is_empty = False
                                append_log(f"    - Passed checks: peak={max_val:.1f}, active_wins={active_wins}/{num_wins} (expected {expected_active_wins})")
                            else:
                                append_log(f"    - Failed active windows check: active_wins={active_wins} < expected {expected_active_wins}")
                        else:
                            append_log(f"    - Failed constant noise check: min_rms/max_rms={min_rms/max_rms:.2f} >= 0.35")
                    else:
                        append_log(f"    - Failed peak check: peak={max_val:.1f} <= 1000.0")
                elif len(words) == 0:
                    is_empty = False  # Empty text block needs no verification
                    append_log("    - Block text contains no words, skipping check.")
                else:
                    append_log("    - No windows available to check.")
                
                if not is_empty:
                    # Convert to pydub segment for further processing
                    segment = pydub.AudioSegment(
                        data=waveform_stripped.tobytes(),
                        sample_width=2,
                        frame_rate=sampling_rate,
                        channels=1
                    )
                    
                    # Custom high-quality trimming
                    # Trim leading silence (keep 50ms)
                    lead_sil_idx = pydub.silence.detect_leading_silence(segment, silence_threshold=-50)
                    lead_sil_idx = max(0, lead_sil_idx - 50)
                    segment = segment[lead_sil_idx:]
                    
                    # Trim trailing silence (keep 300ms to preserve breath)
                    rev_segment = segment.reverse()
                    trail_sil_idx = pydub.silence.detect_leading_silence(rev_segment, silence_threshold=-50)
                    trail_sil_idx = max(0, trail_sil_idx - 300)
                    segment = segment[:-trail_sil_idx] if trail_sil_idx > 0 else segment
                    
                    # Gentle 50ms fade-in and 50ms fade-out
                    segment = segment.fade_in(duration=50).fade_out(duration=50)
                    break
                else:
                    logging.warning(
                        f"Failed energy checks for block {i+1} on attempt {retry+1}/{max_retries}"
                    )
            except Exception as e:
                append_log(f"    - Error during attempt: {e}")
                
        # 3. Ultimate Fallback: Place exact clean silence (no click/noise) if all retries failed
        if segment is None:
            block_dur_ms = block['end_ms'] - block['start_ms']
            append_log(f"  -> WARNING: All attempts failed. Inserting {block_dur_ms}ms of clean silence.")
            segment = pydub.AudioSegment.silent(duration=block_dur_ms, frame_rate=sampling_rate)
            
        # Check length and adjust
        block_dur_ms = block['end_ms'] - block['start_ms']
        
        if stretch_duration:
            # Stretch or speed up to match target_dur_ms
            speed_factor = len(segment) / target_dur_ms
            # Limit the speed adjustment factor to avoid extreme distortion (e.g. 0.6x to 2.0x)
            if 0.6 <= speed_factor <= 2.0:
                sound_altered = segment._spawn(segment.raw_data, overrides={'frame_rate': int(segment.frame_rate * speed_factor)})
                segment = sound_altered.set_frame_rate(segment.frame_rate)
                append_log(f"  -> Stretched segment to match target duration: speed_factor={speed_factor:.3f}")
            elif speed_factor < 0.6:
                sound_altered = segment._spawn(segment.raw_data, overrides={'frame_rate': int(segment.frame_rate * 0.6)})
                segment = sound_altered.set_frame_rate(segment.frame_rate)
                append_log(f"  -> Squeezed segment to max slowdown (0.6x).")
            else:
                sound_altered = segment._spawn(segment.raw_data, overrides={'frame_rate': int(segment.frame_rate * 2.0)})
                segment = sound_altered.set_frame_rate(segment.frame_rate)
                if len(segment) > target_dur_ms:
                    segment = segment[:target_dur_ms]
                append_log(f"  -> Sped up segment to max speedup (2.0x) and truncated to {target_dur_ms}ms.")
            
            # Pad with silence to fit the full block_dur_ms exactly (leaving the end pause)
            pad_len = block_dur_ms - len(segment)
            if pad_len > 0:
                segment = segment + pydub.AudioSegment.silent(duration=pad_len, frame_rate=sampling_rate)
                append_log(f"  -> Padded with {pad_len}ms silence to fit block.")
        else:
            # Natural speed mode
            if len(segment) > block_dur_ms:
                ratio = len(segment) / block_dur_ms
                if ratio > 1.0:
                    ratio = min(ratio, 3.0)
                    sound_altered = segment._spawn(segment.raw_data, overrides={'frame_rate': int(segment.frame_rate * ratio)})
                    segment = sound_altered.set_frame_rate(segment.frame_rate)
                    append_log(f"  -> Sped up segment by {ratio:.3f}x to fit into block.")
            if len(segment) > block_dur_ms:
                segment = segment[:block_dur_ms]
            
        # Overlay at the correct timestamp
        final_audio = final_audio.overlay(segment, position=block['start_ms'])
            
    progress(1.0, desc="Exporting compiled audio...")
    append_log("Exporting compiled audio...")
    yield None, "Exporting final audio...", "\n".join(logs_list)
    
    # Save the final compiled audio to a temporary file
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", dir=os.environ.get("GRADIO_TEMP_DIR"), delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()
    
    try:
        final_audio.export(temp_wav_path, format="wav")
        append_log(f"Successfully generated full audio! Exported to: {temp_wav_path}")
        yield temp_wav_path, "Successfully generated audio from SRT!", "\n".join(logs_list)
    except Exception as e:
        append_log(f"Failed to export final audio: {e}")
        yield None, f"Failed to export final audio: {e}", "\n".join(logs_list)


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:

    sampling_rate = model.sampling_rate

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "Please upload a reference audio."
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="English"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Generation Settings (optional)", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="Speed",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set.",
            )
            du = gr.Number(
                value=None,
                label="Duration (seconds)",
                info=(
                    "Leave empty to use speed."
                    " Set a fixed duration to override speed."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="Inference Steps",
                info="Default: 32. Lower = faster, higher = better quality.",
            )
            dn = gr.Checkbox(
                label="Denoise",
                value=True,
                info="Default: enabled. Uncheck to disable denoising.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Guidance Scale (CFG)",
                info="Default: 2.0.",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt",
                value=True,
                info="apply silence removal and trimming to the reference "
                "audio, add punctuation in the end of reference text (if not already)",
            )
            po = gr.Checkbox(
                label="Postprocess Output",
                value=True,
                info="Remove long silences from generated audio.",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """
# OmniVoice Demo

State-of-the-art text-to-speech model for **600+ languages**, supporting:

- **Voice Clone** — Clone any voice from a reference audio
- **Voice Design** — Create custom voices with speaker attributes
- **SRT Read** — Create custom voices with srt file

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi AI Lab Next-gen Kaldi team.
"""
        )

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Reference Audio / 参考音频",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Recommended: 3–10 seconds audio. "
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label=("Reference Text (optional)" " / 参考音频文本（可选）"),
                            lines=2,
                            placeholder="Transcript of the reference audio. Leave empty"
                            " to auto-transcribe via ASR models.",
                        )
                        vc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        with gr.Accordion("Instruct (optional)", open=False):
                            vc_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vc_subtitle = gr.File(
                            label="Download Subtitle (.srt) / 下载字幕 (.srt)",
                            file_types=[".srt"],
                            interactive=False,
                        )
                        vc_status = gr.Textbox(label="Status / 状态", lines=2)

                def _clone_fn(
                    text, lang, ref_aud, ref_text, instruct, ns, gs, dn, sp, du, pp, po
                ):
                    res = _gen(
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                    )
                    
                    audio_data, status = res
                    if audio_data is None:
                        return None, None, status
                    
                    srt_file_path = None
                    if getattr(model, "_asr_pipe", None) is not None:
                        try:
                            sampling_rate, waveform_int16 = audio_data
                            waveform_float32 = waveform_int16.astype(np.float32) / 32767.0
                            
                            asr_res = model._asr_pipe(
                                {"array": waveform_float32, "sampling_rate": sampling_rate},
                                return_timestamps=True
                            )
                            
                            chunks = asr_res.get("chunks", [])
                            if chunks:
                                srt_content = generate_srt_from_chunks(chunks)
                                
                                temp_srt = tempfile.NamedTemporaryFile(
                                    suffix=".srt", 
                                    dir=os.environ.get("GRADIO_TEMP_DIR"), 
                                    delete=False, 
                                    mode='w', 
                                    encoding='utf-8'
                                )
                                temp_srt.write(srt_content)
                                temp_srt.close()
                                srt_file_path = temp_srt.name
                                status = "Done. Subtitle generated."
                            else:
                                status = "Done. No subtitle chunks detected."
                        except Exception as e:
                            logging.error(f"Error generating subtitle: {e}")
                            status = f"Done. (Failed to generate subtitle: {e})"
                    else:
                        status = "Done. (ASR model not loaded, subtitle generation skipped)"
                        
                    return audio_data, srt_file_path, status

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_subtitle, vc_status],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="Status / 状态", lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )
            # ==============================================================
            # SRT Read / SRT 朗读
            # ==============================================================
            with gr.TabItem("SRT Read"):
                with gr.Row():
                    with gr.Column(scale=1):
                        srt_file = gr.File(
                            label="Upload SRT File / 上传 SRT 文件",
                            file_types=[".srt"],
                        )
                        srt_ref_audio = gr.Audio(
                            label="Reference Audio / 参考音频 (optional for Voice Clone)",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        srt_ref_text = gr.Textbox(
                            label="Reference Text (optional) / 参考音频文本（可选）",
                            lines=2,
                            placeholder="Transcript of the reference audio for voice cloning.",
                        )
                        srt_lang = _lang_dropdown("Language / 语种 (default: English)", value="English")
                        srt_instruct = gr.Textbox(
                            label="Instruct (optional) / Custom voice instructions",
                            lines=2,
                            placeholder="e.g. whisper, male, high pitch",
                        )
                        srt_stretch = gr.Checkbox(
                            label="Khớp khít thời lượng (Stretch to block duration)",
                            value=False,
                            info="If checked, adjusts playback speed of each segment so it fills its SRT timestamp block exactly.",
                        )
                        srt_stretch_pause = gr.Slider(
                            0,
                            1000,
                            value=200,
                            step=50,
                            label="Khoảng nghỉ cuối câu / End-of-sentence pause (ms)",
                            info="Khoảng lặng chừa lại ở cuối mỗi block để tạo khoảng nghỉ tự nhiên giữa các câu (chỉ áp dụng khi bật Khớp khít thời lượng).",
                            
                        )
                        
                        def toggle_stretch_pause(checked):
                            return gr.update(visible=checked)

                        srt_stretch.change(
                            toggle_stretch_pause,
                            inputs=[srt_stretch],
                            outputs=[srt_stretch_pause]
                        )

                        (
                            srt_ns,
                            srt_gs,
                            srt_dn,
                            srt_sp,
                            srt_du,
                            srt_pp,
                            srt_po,
                        ) = _gen_settings()
                        srt_btn = gr.Button("Generate SRT Audio / 朗読 SRT", variant="primary")
                    with gr.Column(scale=1):
                        srt_audio = gr.Audio(
                            label="Final Compiled Audio / 最终合成 audio",
                            type="filepath",
                        )
                        srt_status = gr.Textbox(label="Status / 状态", lines=2)
                        srt_logs = gr.Textbox(
                            label="Execution Logs / Nhật ký chạy",
                            lines=12,
                            max_lines=20,
                            interactive=False,
                            autoscroll=True,
                        )

                def _srt_fn(
                    srt_f, ref_aud, ref_text, lang, instruct, ns, gs, dn, sp, du, pp, po, stretch, stretch_pause
                ):
                    if srt_f is None:
                        yield None, "Please upload an SRT file.", ""
                        return
                    srt_path = srt_f.name if hasattr(srt_f, 'name') else str(srt_f)
                    
                    for audio, status, logs in synthesize_srt(
                        srt_file_path=srt_path,
                        model=model,
                        sampling_rate=sampling_rate,
                        language=lang,
                        ref_audio=ref_aud,
                        ref_text=ref_text,
                        instruct=instruct,
                        num_step=ns,
                        guidance_scale=gs,
                        denoise=dn,
                        speed=sp,
                        preprocess_prompt=pp,
                        postprocess_output=po,
                        stretch_duration=stretch,
                        stretch_pause_ms=stretch_pause
                    ):
                        yield audio, status, logs

                srt_btn.click(
                    _srt_fn,
                    inputs=[
                        srt_file,
                        srt_ref_audio,
                        srt_ref_text,
                        srt_lang,
                        srt_instruct,
                        srt_ns,
                        srt_gs,
                        srt_dn,
                        srt_sp,
                        srt_du,
                        srt_pp,
                        srt_po,
                        srt_stretch,
                        srt_stretch_pause,
                    ],
                    outputs=[srt_audio, srt_status, srt_logs],
                )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    # Register allowed paths for serving files inline (avoiding attachment headers)
    allowed_dirs = [
        os.path.abspath(os.getcwd()),
        os.environ.get("GRADIO_TEMP_DIR", "")
    ]
    allowed_dirs = list(set(d for d in allowed_dirs if d))

'''
    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
        allowed_paths=allowed_dirs,
    )
    '''
    demo.queue().launch(
        share=True
        allowed_paths=allowed_dirs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
