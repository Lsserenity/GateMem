# PRML

## 文件结构
```
PRML/
├─ titans.py
├─ memory_network.py
├─ models.py
├─ README.md
├─ config.yaml                 # ✅ 用户填写的配置文件放这里
├─ config_loader.py            # ✅ 读取/解析 config.yaml
└─ dynamic_cheatsheet/         # ✅ DC层代码都放这个包里
   ├─ __init__.py
   ├─ schema.py                # LLM 输出条目结构（可选，但建议）
   ├─ llm_client.py            # LLM 接口封装（先写 stub，后面接 qwen/openai）
   ├─ embed_client.py          # embedding 接口封装（先写 stub，后面接真实 embedding）
   └─ cheatsheet_memory.py     # ✅ DC 核心：存储/检索/更新 -> dc_memory
```

## 库需求
```
pip install pyyaml
```

## 用户须配置

* .env
```
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

* config.yaml