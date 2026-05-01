# HuBERTone

将人类歌声中的音色映射到可计算的向量空间——无需原始编码器，仅凭现有码本与自监督特征，就能为 Synthesizer V 生成可用的音色嵌入。项目中自带开箱即用的模型与 WebUI。

## 解决问题

在歌声合成（如 Synthesizer V）中扩展新音色时，通常依赖难以获取的原始声码器或编码器，大大限制了从海量无标注人声录音中直接提取并利用音色的可能。

## 项目结构

```
hubert/
├── inference.py      # 公共推理逻辑（滑窗、聚合、嵌入提取）
├── model.py          # HuBERT + 注意力池化 + MLP 模型定义
├── data.py           # 训练数据集（滑窗分段、缓存）
├── train.py          # 训练脚本
├── infer.py          # 命令行推理工具
├── ap.py             # Gradio WebUI
├── utils.py          # 工具函数
└── checkpoints/      # 模型权重存放目录
```

## 安装依赖

```bash
pip install torch torchaudio transformers gradio numpy tqdm
```

## 训练

### 数据准备

1. 将 WAV 音频文件放入 `wav/` 目录
2. 准备 `emb_list.json`，格式如下：
   ```json
   [
     {
       "version": "音色版本ID",
       "data": "32维向量的hex编码（256字符）",
       "l2_norm": 0.77,
       "algorithm": "gaussian",
       "gender": 1.0
     }
   ]
   ```

### 开始训练

```bash
python -m hubert.train --wav_dir wav --emb_list emb_list.json
```

常用参数：
- `--layer_idx 8`：使用 HuBERT 第 8 层特征
- `--window_sec 8.0`：训练窗口长度（秒）
- `--hop_sec 4.0`：滑窗步长（秒）
- `--num_epochs 20`：训练轮数
- `--batch_size 32`：批大小
- `--save_path checkpoints/hubert_sv_model.pth`：模型保存路径

## 推理

### 命令行

```bash
python -m hubert.infer --model_path checkpoints/hubert_sv_model.pth --input_wav input.wav
```

输出：
- 32 维音色嵌入向量
- Hex 编码格式

### WebUI

```bash
python -m hubert.ap --model_path checkpoints/hubert_sv_model.pth
```

启动后访问 `http://localhost:7860`，上传音频即可获取音色嵌入。

WebUI 参数说明：
- `window_sec`：滑窗大小（秒）
- `hop_sec`：滑窗步长（秒）
- `min_coverage`：最后一个窗口的最小覆盖率
- `min_rms_ratio`：最小 RMS 能量比（过滤静音）
- `gauss_sigma_ratio`：高斯加权的 sigma 系数
- `scale`：输出嵌入的缩放因子
- `direct_average`：是否跳过方向归一化直接加权平均

## 模型架构

1. **特征提取**：冻结的 HuBERT (`facebook/hubert-base-ls960`) 第 8 层
2. **池化**：注意力统计池化（Attentive Statistics Pooling）
3. **预测头**：残差 MLP，输出 32 维方向向量 + 1 维幅度标量

## License

See [LICENSE](LICENSE) for details.
