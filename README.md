# STFTCodec: 基于STFT的高效音频编解码器

一个基于短时傅里叶变换(STFT)的神经音频编解码器，支持高质量音频压缩和重建。该项目实现了多个版本的编解码器模型，提供不同的压缩率和频谱表示方法。

## 📋 项目概述

STFTCodec是一个现代化的音频编解码器，使用神经网络技术实现高效的音频压缩。项目支持多种配置和版本：

- **v1版本**: 基础版本，压缩率为320
- **v2版本**: 改进版本，压缩率提升至1920，显著提高了压缩效率
- **v361**: 使用幅度谱和相位谱
- **v3631**: 使用幅度谱、相位谱和相位展开差分谱

## 🚀 主要特性

- 🎵 **高质量音频压缩**: 支持24kHz采样率的音频编解码
- 📊 **多种频谱表示**: 支持不同的频谱分析方法
- 🔧 **灵活配置**: 丰富的超参数配置选项
- ⚡ **高效训练**: 支持GPU加速和分布式训练
- 🎯 **多种损失函数**: 包含Mel谱损失、多尺度STFT损失、对抗损失等
- 📈 **可视化监控**: 集成TensorBoard支持

## 📁 项目结构

```
├── apcodec/                    # 模型结构代码
│   ├── dac_layer.py           # DAC层实现
│   ├── models_apcodec.py      # 基础模型和损失函数
│   ├── models_stftcodecV2_v361_h40r48_r344_keepRB.py  # v2版本模型
│   ├── models_stftcodec_v361_numLayer_len_1211.py    # v361版本模型
│   ├── models_stftcodec_v3631_ED_1205.py              # v3631版本模型
│   ├── models_dac_Disc.py     # 判别器模型
│   ├── mstft.py               # 多尺度STFT实现
│   └── mpd.py                 # 多周期判别器
├── config/                     # 配置文件
│   ├── v2/                    # v2版本配置
│   └── stftcodec/             # STFT编解码器配置
├── datasets/                   # 数据集相关代码
├── common/                     # 通用工具函数
├── metrics/                    # 评估指标
├── results/                    # 训练结果保存目录
├── quantize_valCodeUsage_mulSize.py  # 量化实现
└── 训练和推理脚本
```

## 🛠️ 安装要求

```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install numpy scipy librosa soundfile
pip install tensorboard tqdm pyyaml
pip install einops matplotlib
```

## 📚 训练脚本说明

### 主要训练脚本

1. **`train_stftcodec_v361_len15960_numLayer_dacDisc_speed.py`**
   - v361版本训练脚本
   - 使用幅度谱和相位谱
   - 支持DAC判别器训练

2. **`train_stftcodec_v3631_timeDiff_dacDisc_speed_1215.py`**
   - v3631版本训练脚本
   - 使用幅度谱、相位谱和相位展开差分谱
   - 包含时域差分损失

3. **`train_stftcodecV2_v361_h40r48_344_keepRB_0502.py`**
   - v2版本训练脚本
   - 压缩率提升至1920
   - 保持残差块结构

### 版本说明

- **v361**: 仅使用幅度谱和相位谱进行编码
- **v3631**: 使用幅度谱、相位谱、相位展开差分谱，提供更丰富的频域信息
- **v2**: 在v1基础上进一步提高压缩率，从320提升到1920

## 🏃‍♂️ 快速开始

### V2版本训练

```bash
CUDA_VISIBLE_DEVICES=0 python train_stftcodecV2_v361_h40r48_344_keepRB_0502.py --config ./config/v2/baseline.yml --checkpoint_path results/v2/

CUDA_VISIBLE_DEVICES=0 python train_stftcodec_v361_len15960_numLayer_dacDisc_speed.py --config ./config/v2/baseline.yml --checkpoint_path results/nov2/

CUDA_VISIBLE_DEVICES=0 python train_stftcodec_v3631_timeDiff_dacDisc_speed_1215.py
--config ./config/v2/baseline.yml --checkpoint_path results/v3631/

```

### 推理使用

```bash
# V2版本推理
CUDA_VISIBLE_DEVICES=6 python get_inference_audio_stftcodecV2_v361_0427.py \
    --config ./results/v2/your_model/config.yml \
    --reference_dir /path/to/input/audio/ \
    --output_dir /path/to/output/ \
    --encoder_ckpt ./results/v2/your_model/encoder_ckpt.pth \
    --decoder_ckpt ./results/v2/your_model/decoder_ckpt.pth
```

### V361版本推理

```bash
python get_inference_audio_stftcodec_v361.py \
    --config /path/to/config.yml \
    --input_dir /path/to/input/ \
    --output_dir /path/to/output/
```

### V3631版本推理

```bash
python get_inference_audio_stftcodec_v3631_timeDiff_libriTTS.py \
    --config /path/to/config.yml \
    --input_dir /path/to/input/ \
    --output_dir /path/to/output/
```

## ⚙️ 配置参数

### 主要模型参数

- `sampling_rate`: 采样率 (24000Hz)
- `n_fft`: FFT点数 (1024)
- `hop_size`: 跳跃大小 (40)
- `win_size`: 窗口大小 (320)
- `n_codebooks`: 码本数量 (6)
- `codebook_size`: 码本大小 (4096)
- `encoder_dim`: 编码器维度 (256)

### 训练参数

- `batch_size_train`: 训练批次大小 (32)
- `learning_rate_g`: 生成器学习率 (0.00005)
- `learning_rate_d`: 判别器学习率 (0.00005)
- `training_epochs`: 训练轮数 (1000)

## 🔧 量化技术

项目使用了先进的向量量化技术：

- **因式分解码**: 在低维空间进行最近邻查找，提高码本利用率
- **L2正则化码**: 将欧氏距离转换为余弦相似度，改善训练稳定性
- **码本利用率统计**: 实时监控码本的使用情况

详细实现请参考 `quantize_valCodeUsage_mulSize.py`。

## 📊 损失函数

项目集成了多种损失函数：

1. **Mel谱损失**: 确保重建音频的感知质量
2. **多尺度STFT损失**: 保持频域特征的一致性
3. **对抗损失**: 通过判别器提升生成质量
4. **向量量化损失**: 优化码本学习
5. **SI-SDR损失**: 信号失真比损失

## 🎯 模型架构

### 编码器
- 使用ResidualUnit和EncoderBlock构建
- 支持多层卷积和残差连接
- 集成Snake激活函数

### 解码器
- 对称的上采样结构
- DecoderBlock实现特征重建
- 动态输出填充计算

### 判别器
- 多周期判别器(MPD)
- 多分辨率判别器(MRD)
- 多尺度判别器(MSD)支持

## 📈 性能监控

使用TensorBoard监控训练过程：

```bash
tensorboard --logdir results/your_experiment/logs
```

监控指标包括：
- 生成器和判别器损失
- Mel谱损失变化
- 码本利用率
- 音频质量指标

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目。在提交代码前，请确保：

1. 代码符合项目风格
2. 添加必要的注释和文档
3. 通过基本的测试验证

## 📄 许可证

本项目采用 [MIT许可证](LICENSE)。

## 🙏 致谢

感谢所有为神经音频编解码器领域做出贡献的研究者和开发者。

---

**注意**: 训练需要大量的计算资源和时间，建议使用GPU加速。推理过程相对较快，可以实时处理音频文件。 # Mel_cap_demo
