# 代码目录说明

./data_reverse_dropvowel 目录存放生成数据的.json 文件<br>
./dynamic_cheatsheet 是 dynamic_cheatsheet 的模块<br>
./minGPT 是整个语言模型的部分，其核心部分来自https://github.com/karpathy/minGPT<br>
./minGPT/mingpt/memory_network.py 是添加的 neural memory 模块<br>
./minGPT/mingpt/model.py 是 GPT 的核心模块，通过修改注入了添加的 memory 逻辑<br>

download.py 用于从 Hugging face 下载预训练好的 GPT2 的模型参数，便于迁移到服务器上使用<br>
test_no_memory_sample.ipynb, test_no_memory.ipynb 分别实现，不带 memory 逻辑的两种训练方式。<br>
test_no_dc_sample.ipynb, test_no_dc.ipynb 分别实现，不带 dynamic cheatsheet 逻辑的两种训练方式。<br>
test_sample.ipynb, test.ipynb 分别实现，dc + nm 的两种训练方式。<br>

# 依赖的环境

实验依赖的环境，导出为 environment.yml，使用时直接 conda env create -f environment.yml 即可

# 运行说明

1. 直接运行所有的 .ipynb 文件即可实现模型的训练和测试
2. 由于调用了 hugging face 的模型参数，所以这一句需要做相应的修改：
   ```python
   model, _ = GPT.from_pretrained(model_type, types="nm", model_dir="gpt2_local")
   ```
   如果需要直接从 hugging face 加载，model_dir=None 即可
   否则，需要先 pip install huggingface_hub 然后运行 download.py，将模型参数下载到本地。model_dir 需要设置成参数文件夹的相对路径。
3. 由于 dynamic_cheatsheet 3. 模块调用了千问的 API， 所以 test_sample.ipynb, test.ipynb 需要在代码开头配置 API key 为环境变量。
