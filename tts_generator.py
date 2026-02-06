"""
TTS Generator

封装 IndexTTS2，提供简洁的 API 接口用于文本转语音。
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Optional, List, Dict
import threading
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 导入 IndexTTS2
# 方式1: 如果通过 pip 安装（推荐）
# pip install git+https://github.com/index-tts/index-tts.git
try:
    from indextts.infer_v2 import IndexTTS2
except ImportError:
    # 方式2: 如果使用 Git submodule
    # git submodule add https://github.com/index-tts/index-tts.git submodules/index-tts
    project_root = Path(__file__).parent.parent
    submodule_path = project_root / "submodules" / "index-tts"
    if submodule_path.exists():
        sys.path.insert(0, str(submodule_path))
        from indextts.infer_v2 import IndexTTS2
    else:
        raise ImportError(
            "无法导入 IndexTTS2。\n"
            "请选择以下方式之一安装：\n"
            "1. 通过 pip 安装（推荐）: pip install git+https://github.com/index-tts/index-tts.git\n"
            "2. 使用 Git submodule: git submodule add https://github.com/index-tts/index-tts.git submodules/index-tts"
        )

try:
    from pydub import AudioSegment
except ImportError:
    raise ImportError("请安装 pydub: pip install pydub")


class TTSGenerator:
    """
    文本转语音生成器
    
    封装 IndexTTS2，提供简洁的 API 接口
    """
    
    def __init__(
        self,
        model_dir: str = "models/checkpoints",
        use_fp16: bool = True,
        use_deepspeed: bool = False,
        use_cuda_kernel: bool = True
    ):
        """
        初始化 TTS 生成器
        
        Args:
            model_dir: 模型文件目录路径
            use_fp16: 是否使用 FP16 加速（需要 GPU 支持）
            use_deepspeed: 是否使用 DeepSpeed 加速
            use_cuda_kernel: 是否使用 CUDA 内核加速
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"模型目录不存在: {model_dir}\n"
                "请从 IndexTTS2 官方仓库下载模型文件。"
                "\n参考: https://github.com/index-tts/index-tts"
            )
        
        config_path = os.path.join(model_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"配置文件不存在: {config_path}\n"
                "请确保模型目录包含 config.yaml 文件。"
            )
        
        self.model_dir = model_dir
        self.tts = IndexTTS2(
            model_dir=model_dir,
            cfg_path=config_path,
            use_fp16=use_fp16,
            use_deepspeed=use_deepspeed,
            use_cuda_kernel=use_cuda_kernel,
        )


    def analyze_text(self, text: str, max_text_tokens_per_segment: int = 120):
        """
        分析文本，预览分句信息
        
        Args:
            text: 要分析的文本
            max_text_tokens_per_segment: 每段最大 token 数
            
        Returns:
            dict: 包含分句信息的字典
                - total_tokens: 总 token 数
                - segment_count: 分段数量
                - segments: 分段列表，每个元素包含 index, tokens, text
        """
        if not text or not text.strip():
            return {
                "total_tokens": 0,
                "segment_count": 0,
                "segments": []
            }
        
        text_tokens_list = self.tts.tokenizer.tokenize(text)
        segments = self.tts.tokenizer.split_segments(
            text_tokens_list,
            max_text_tokens_per_segment=max_text_tokens_per_segment
        )
        
        return {
            "total_tokens": len(text_tokens_list),
            "segment_count": len(segments),
            "segments": [
                {
                    "index": i,
                    "tokens": len(s),
                    "text": ''.join(s)
                }
                for i, s in enumerate(segments)
            ]
        }
    
    def generate(
        self,
        text: str,
        voice_reference: str,
        output_path: str,
        output_format: str = "mp3",
        max_text_tokens_per_segment: Optional[int] = None,
        emo_audio_prompt: Optional[str] = None,
        emo_text: Optional[str] = None,
        emo_alpha: float = 1.0,
        verbose: bool = False
    ) -> str:
        """
        生成语音
        
        Args:
            text: 要转换的文本
            voice_reference: 音色参考音频路径
            output_path: 输出文件路径
            output_format: 输出格式 ("mp3" 或 "wav")
            max_text_tokens_per_segment: 每段最大 token 数（None 则自动计算）
            emo_audio_prompt: 情感参考音频路径（可选，与 emo_text 二选一）
            emo_text: 情感文本描述（可选，与 emo_audio_prompt 二选一）
                      例如："开心"、"悲伤"、"愤怒"、"惊讶"等
            emo_alpha: 情感权重（0.0-1.0）
            verbose: 是否显示详细信息
            
        Returns:
            输出文件路径
            
        Raises:
            FileNotFoundError: 如果音色参考文件不存在
            RuntimeError: 如果音频格式转换失败
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")
        
        if not os.path.exists(voice_reference):
            raise FileNotFoundError(f"音色参考文件不存在: {voice_reference}")
        
        if emo_audio_prompt and not os.path.exists(emo_audio_prompt):
            raise FileNotFoundError(f"情感参考文件不存在: {emo_audio_prompt}")
        
        # emo_audio_prompt 和 emo_text 不能同时使用
        if emo_audio_prompt and emo_text:
            raise ValueError("emo_audio_prompt 和 emo_text 不能同时使用，请选择其中一种方式")
        
        # 自动计算 max_text_tokens_per_segment
        if max_text_tokens_per_segment is None:
            text_length = len(text)
            if text_length < 500:
                max_text_tokens_per_segment = 120
            elif text_length < 2000:
                max_text_tokens_per_segment = 150
            else:
                max_text_tokens_per_segment = 180
            
            # 确保不超过模型限制
            max_allowed = (
                self.tts.cfg.gpt.max_text_tokens
                if hasattr(self.tts.cfg.gpt, 'max_text_tokens')
                else 200
            )
            max_text_tokens_per_segment = min(
                max_text_tokens_per_segment,
                max_allowed
            )
        
        # 生成临时 wav 文件
        temp_wav = output_path.replace(f".{output_format}", "_temp.wav")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 生成音频
        infer_kwargs = {
            "spk_audio_prompt": voice_reference,
            "text": text,
            "output_path": temp_wav,
            "emo_alpha": emo_alpha,
            "verbose": verbose,
            "max_text_tokens_per_segment": max_text_tokens_per_segment,
        }
        
        # 添加情感参数（优先使用 emo_text，如果提供的话）
        if emo_text:
            infer_kwargs["use_emo_text"] = True
            infer_kwargs["emo_text"] = emo_text
            print(f"  传递 emo_text 到 IndexTTS2: {emo_text}")
        elif emo_audio_prompt:
            infer_kwargs["emo_audio_prompt"] = emo_audio_prompt
            print(f"  传递 emo_audio_prompt 到 IndexTTS2: {emo_audio_prompt}")
        
        try:
            self.tts.infer(**infer_kwargs)
        except TypeError as e:
            # 如果 IndexTTS2 不支持 emo_text 参数，直接报错退出
            if "emo_text" in str(e) or "unexpected keyword" in str(e).lower():
                raise RuntimeError(
                    f"IndexTTS2 不支持 emo_text 参数: {e}\n"
                    "请检查 IndexTTS2 版本是否支持情感文本输入。"
                ) from e
            else:
                raise
        
        # 转换为指定格式
        if output_format.lower() == "mp3":
            try:
                audio = AudioSegment.from_wav(temp_wav)
                audio.export(output_path, format="mp3", bitrate="192k")
                # 删除临时文件
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            except Exception as e:
                # 如果转换失败，保留 wav 文件
                if os.path.exists(temp_wav):
                    fallback_path = output_path.replace(".mp3", ".wav")
                    os.rename(temp_wav, fallback_path)
                raise RuntimeError(
                    f"MP3 转换失败: {e}\n"
                    "请确保已安装 ffmpeg: sudo apt-get install ffmpeg"
                )
        else:
            # 直接使用 wav
            if temp_wav != output_path:
                os.rename(temp_wav, output_path)
        
        return output_path
    
    def _add_background_music(
        self,
        audio: AudioSegment,
        background_music: str,
        bgm_volume: float = 0.3,
        verbose: bool = False
    ) -> AudioSegment:
        """
        为音频添加背景音乐
        
        前3秒：只有背景音乐，音量从1.0倍渐变到0.3倍
        3秒后：开始语音，背景音乐保持0.3倍
        
        Args:
            audio: 原始音频（AudioSegment 对象）
            background_music: 背景音乐文件路径
            bgm_volume: 背景音乐音量（0.0-1.0，默认 0.3，用于语音部分）
            verbose: 是否显示详细信息
            
        Returns:
            混合后的音频（AudioSegment 对象），总长度 = 3秒（纯背景音乐）+ 语音长度（带背景音乐）
            
        Raises:
            FileNotFoundError: 如果背景音乐文件不存在
        """
        if not os.path.exists(background_music):
            raise FileNotFoundError(f"背景音乐文件不存在: {background_music}")
        
        print(f"正在添加背景音乐: {background_music}")
        
        try:
            from pydub.effects import normalize
            
            # 加载背景音乐
            bgm = AudioSegment.from_file(background_music)
            
            cover_duration_ms = 3000  # 封面时长 3 秒（毫秒）
            cover_high_volume = 1.0  # 封面高音量倍数
            
            # 计算总长度：3秒（纯背景音乐）+ 语音长度（带背景音乐）
            total_duration_ms = cover_duration_ms + len(audio)
            
            # 如果背景音乐比总长度短，循环播放
            if len(bgm) < total_duration_ms:
                num_loops = int(total_duration_ms / len(bgm)) + 1
                bgm = bgm * num_loops
            
            # 裁剪背景音乐到总长度
            bgm_full = bgm[:total_duration_ms]
            
            # 处理前3秒：音量从1.5倍渐变到0.3倍
            cover_bgm = bgm_full[:cover_duration_ms]
            cover_segments = []
            segment_duration_ms = 100  # 每100ms一个片段，实现平滑渐变
            
            for i in range(0, cover_duration_ms, segment_duration_ms):
                segment_end = min(i + segment_duration_ms, cover_duration_ms)
                segment = cover_bgm[i:segment_end]
                
                # 计算当前时间点的音量倍数
                t_seconds = i / 1000.0  # 当前时间（秒）
                if t_seconds < 2.0:
                    # 前2秒：保持1.5倍高音量
                    volume_multiplier = cover_high_volume
                elif t_seconds < 3.0:
                    # 2-3秒：从1.5倍线性渐弱到0.3倍
                    fade_progress = (t_seconds - 2.0) / 1.0
                    volume_multiplier = cover_high_volume - (cover_high_volume - bgm_volume) * fade_progress
                else:
                    # 3秒：保持0.3倍
                    volume_multiplier = bgm_volume
                
                # 应用音量调整
                # pydub 使用 dB 调整，volume_multiplier 转换为 dB
                # 例如：1.5倍 = +3.5dB, 0.3倍 = -10.5dB
                volume_db = 20 * (volume_multiplier - 1.0)  # 线性到dB的近似转换
                segment = segment + volume_db
                cover_segments.append(segment)
            
            # 拼接封面背景音乐
            cover_audio = sum(cover_segments)
            
            # 处理正文部分：背景音乐保持0.3倍，与语音混合
            body_bgm = bgm_full[cover_duration_ms:]
            body_bgm = body_bgm[:len(audio)]  # 确保长度匹配
            # 调整正文部分背景音乐音量为0.3倍
            body_bgm_volume_db = 20 * (bgm_volume - 1.0)  # bgm_volume=0.3 时约为 -14dB
            body_bgm = body_bgm + body_bgm_volume_db
            
            # 混合正文部分的背景音乐和语音
            body_audio = audio.overlay(body_bgm)
            
            # 拼接：封面（纯背景音乐）+ 正文（语音+背景音乐）
            mixed_audio = cover_audio + body_audio
            
            print(f"背景音乐已添加，封面3秒音量渐变（1.0倍→0.3倍），正文部分音量: {bgm_volume * 100:.0f}%")
            
            return mixed_audio
        except Exception as e:
            print(f"警告: 添加背景音乐失败: {e}")
            print("继续使用无背景音乐的版本")
            return audio
    
    def get_voice_reference(
        self,
        speaker: str,
        base_dir: Path,
        role_voice_map: Dict[str, str]
    ) -> str:
        """
        根据角色名获取音色参考文件路径
        
        Args:
            speaker: 角色名称
            base_dir: 音色文件基础目录
            role_voice_map: 角色到音色文件名的映射字典
            
        Returns:
            音色参考文件的完整路径
            
        Raises:
            FileNotFoundError: 如果音色文件不存在
        """
        speaker = speaker.strip()
        if speaker in role_voice_map:
            voice_file = role_voice_map[speaker]
        else:
            voice_file = role_voice_map["Narrator"]
        
        voice_path = None
        voice_file_path = Path(str(voice_file))
        if voice_file_path.is_absolute():
            voice_path = voice_file_path
        elif "/" in str(voice_file) or "\\" in str(voice_file):
            # Treat as repo-relative path if it looks like a path
            candidate = (Path.cwd() / voice_file_path).resolve()
            voice_path = candidate
        else:
            voice_path = base_dir / voice_file
        
        # 验证音色文件是否存在
        if not voice_path.exists():
            raise FileNotFoundError(
                f"角色 '{speaker}' 的音色文件不存在\n"
                f"  文件路径: {voice_path}\n"
                f"  基础目录: {base_dir}\n"
                f"  音色文件名: {voice_file}"
            )
        
        return str(voice_path)
    
    def generate_from_json(
        self,
        json_data: List[Dict[str, str]],
        output_path: str,
        timestamps_file: str,
        output_format: str = "mp3",
        silence_duration: float = 0.3,
        background_music: Optional[str] = None,
        bgm_volume: float = 0.3,
        verbose: bool = False,
        casting_info_file: Optional[str] = None,
    ) -> str:
        """
        从 JSON 结构化数据生成多角色语音
        
        Args:
            json_data: JSON 结构化数据，每个元素包含 "speaker" 和 "zh" 字段
            output_path: 输出文件路径
            output_format: 输出格式 ("mp3" 或 "wav")
            silence_duration: 片段之间的静音时长（秒）
            background_music: 背景音乐文件路径（可选，支持 mp3, wav, m4a 等格式）
            bgm_volume: 背景音乐音量（0.0-1.0，默认 0.3，即 30% 音量）
            verbose: 是否显示详细信息
            casting_info_file: 选角信息文件路径（可选，如果为 None，则从输出目录查找 casting_info.json）
            timestamps_file: 时间戳输出路径（必填）
            
        Returns:
            输出文件路径
        """
        from pydub import AudioSegment
        import json as json_lib
        from pathlib import Path
        
        # 加载选角信息        
        casting_info_path = Path(casting_info_file)
        if not casting_info_path.exists():
            raise FileNotFoundError(
                f"选角信息文件不存在: {casting_info_file}\n"
                "请确保 CastingAgent 已保存选角信息到输出目录。"
            )
        
        with open(casting_info_path, 'r', encoding='utf-8') as f:
            casting_info = json_lib.load(f)
        
        base_dir = Path(casting_info.get("base_dir", "assets/ref_voice"))
        role_voice_map = casting_info.get("role_voice_map", {})
                
        audio_segments = []
        temp_files = []
        timestamps = []  # 记录每个片段的时间戳
        current_time = 0.0  # 当前累计时间
        
        try:
            for i, item in enumerate(json_data):
                speaker = item.get("speaker", "Narrator")
                text = item.get("zh", "").strip()
                emo = item.get("emo", "").strip()
                
                if not text:
                    continue
                
                # 获取角色对应的音色参考
                voice_ref = self.get_voice_reference(speaker, base_dir, role_voice_map)
                
                # 如果不是 Narrator，使用 emo_text（如果提供）
                emo_text = None
                if speaker != "Narrator":
                    emo_text = emo
                    print(f"  使用情感: {emo_text}")
                
                emotion_info = f" (情感: {emo})" if emo_text else ""
                print(f"生成片段 {i+1}/{len(json_data)}: {speaker}{emotion_info}")
                
                # 记录片段开始时间
                segment_start_time = current_time
                
                # 生成临时音频文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    temp_wav = tmp_file.name
                    temp_files.append(temp_wav)
                
                # 生成语音
                self.generate(
                    text=text,
                    voice_reference=voice_ref,
                    output_path=temp_wav,
                    output_format="wav",
                    emo_text=emo_text,
                    emo_alpha=0.4,
                    verbose=verbose
                )
                
                # 加载音频片段
                segment = AudioSegment.from_wav(temp_wav)
                # 归一化音量，确保所有片段音量一致
                segment = segment.normalize()
                
                # 添加淡入淡出，避免拼接时的"click"声
                # 使用 20ms 的淡入淡出，足够平滑边界又不会影响语音内容
                fade_duration = 20  # 毫秒
                if len(segment) > fade_duration * 2:
                    segment = segment.fade_in(fade_duration).fade_out(fade_duration)
                
                # 计算片段时长（秒）- 使用处理后的实际音频长度，确保时间戳准确
                # 这样可以准确反映实际播放时长
                segment_duration = len(segment) / 1000.0
                segment_end_time = segment_start_time + segment_duration
                
                # 记录时间戳信息
                timestamps.append({
                    "index": len(timestamps),
                    "speaker": speaker,
                    "text": text,
                    "start_time": segment_start_time,
                    "end_time": segment_end_time,
                    "duration": segment_duration
                })
                
                audio_segments.append(segment)
                current_time = segment_end_time
                
                # 添加静音间隔（除了最后一个片段）
                if i < len(json_data) - 1:
                    silence = AudioSegment.silent(duration=int(silence_duration * 1000))
                    audio_segments.append(silence)
                    current_time += silence_duration
            
            # 拼接所有音频片段
            print("正在拼接音频片段...")
            
            final_audio = sum(audio_segments)
            
            # 记录纯语音音频的时长（用于验证时间戳）
            pure_audio_duration = len(final_audio) / 1000.0
            last_timestamp_end = timestamps[-1]['end_time'] if timestamps else 0.0
            print(f"纯语音音频时长: {pure_audio_duration:.3f} 秒")
            print(f"最后一个时间戳结束时间: {last_timestamp_end:.3f} 秒")
            if abs(pure_audio_duration - last_timestamp_end) > 0.1:
                print(f"⚠️  警告: 时间戳总时长 ({last_timestamp_end:.3f}秒) 与音频实际时长 ({pure_audio_duration:.3f}秒) 不匹配，差值: {abs(pure_audio_duration - last_timestamp_end):.3f}秒")
            
            # 添加背景音乐（如果提供）
            if background_music:
                final_audio = self._add_background_music(
                    final_audio,
                    background_music,
                    bgm_volume,
                    verbose
                )
            
            # 保存最终音频
            if output_format.lower() == "mp3":
                final_audio.export(output_path, format="mp3", bitrate="192k")
            else:
                final_audio.export(output_path, format="wav")
            
            # 保存时间戳文件
            if not timestamps_file:
                raise ValueError("timestamps_file 不能为空")
            timestamps_path = Path(timestamps_file)
            timestamps_path.parent.mkdir(parents=True, exist_ok=True)
            with open(timestamps_path, 'w', encoding='utf-8') as f:
                json_lib.dump(timestamps, f, ensure_ascii=False, indent=2)
            print(f"✓ 时间戳文件已保存: {timestamps_path}")
            
            return output_path
            
        finally:
            # 清理临时文件
            for tmp_file in temp_files:
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except:
                        pass


_TTS_GENERATOR_INSTANCE: Optional[TTSGenerator] = None
_TTS_GENERATOR_LOCK = threading.Lock()


def get_tts_generator(model_dir: str = "models/checkpoints") -> TTSGenerator:
    """
    获取全局单例的 TTSGenerator，避免重复加载大模型
    """
    global _TTS_GENERATOR_INSTANCE
    if _TTS_GENERATOR_INSTANCE is not None:
        return _TTS_GENERATOR_INSTANCE

    with _TTS_GENERATOR_LOCK:
        if _TTS_GENERATOR_INSTANCE is None:
            _TTS_GENERATOR_INSTANCE = TTSGenerator(model_dir=model_dir)
    return _TTS_GENERATOR_INSTANCE

