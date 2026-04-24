# Resume Recognition

面向非结构化 PDF 简历的端到端信息抽取与岗位匹配流水线：  
**PDF 文本提取 -> BERT NER 实体识别 -> 技能标准化 -> 岗位匹配排序**。

该项目适合作为课程项目、技术演示或简历中的 AI 系统实践案例，重点在于从模型训练到工程串联的完整闭环。

## Features

- **End-to-end pipeline**：一条命令完成简历处理与结果导出
- **BERT-based NER**：基于 BIO 标注识别姓名、技能、经历等实体
- **Skill normalization**：词表、别名与模糊匹配结合，降低技能写法噪声
- **Job matching**：基于标准化技能与岗位索引计算重叠并排序
- **Demo-ready outputs**：按简历生成结构化 JSON，便于可视化和汇报

## Project Structure

- `run_pipeline.py`：端到端主入口
- `pipeline/input/`：待处理 PDF 输入目录
- `pipeline/output/`：流水线输出目录（entities/normalized/job_matches）
- `ner/`：NER 模型训练与推理代码（`train.py`、`test.py`、`server/utils.py`）
- `skill_normalizer/`：技能标准化逻辑与规则
- `job_ingestion/`：岗位数据清洗与导出（如 `job_ingestion/output/job_postings_extracted.jsonl`）

## Requirements

推荐环境：

- Python `3.9`
- `pip`（建议升级到较新版本）

安装依赖（在仓库根目录执行）：

```bash
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

说明：

- 根目录 `requirements.txt` 已包含端到端运行所需依赖（NER 推理/训练、技能标准化、API 相关包）。
- 仓库中还保留了 `NER/requirements.txt`（历史拆分文件，内容与核心依赖基本一致）。

## Quick Start

1. 进入仓库根目录。
2. 准备模型权重：`ner/model/model-state.bin`（推荐路径）。
   - 若仅有根目录 `model-state.bin`，`run_pipeline.py` 会自动回退加载。
3. 将待处理简历 PDF 放入 `pipeline/input/`。
4. 运行流水线：

```bash
python run_pipeline.py
```

5. 在 `pipeline/output/` 查看结果文件：
   - `<简历名>_entities.json`：实体识别结果（合并后）
   - `<简历名>_normalized.json`：技能标准化结果与审计信息
   - `<简历名>_job_matches.json`：岗位匹配结果（依赖岗位索引数据）

## Train NER Model

在仓库根目录运行：

```bash
python ner/train.py
```

或在 `ner/` 下运行：

```bash
python3 train.py
```

快速推理测试：

```bash
python ner/test.py
```

测试会读取 `ner/demo/` 示例 PDF，并在同目录输出 `demo_entities_*.json`。

## Data and Demo

- 示例文件：
  - [Demo Resume PDF](ner/demo/Resume%20-%20Ayush%20Srivastava.pdf)
  - [Demo JSON Response](ner/demo/response.json)
- 原始标注数据集（Kaggle）：
  - [Resume Entities for NER](https://www.kaggle.com/dataturks/resume-entities-for-ner)

## Notes

- 当前仓库聚焦可运行代码与示例数据，文档以本 `README.md` 为主。
- 如用于课程答辩，建议结合 `pipeline/output/` 的结果文件展示端到端效果。
