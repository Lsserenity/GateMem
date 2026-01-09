整个架构在 minGPT
dc 逻辑在 minGPT/mingpt/model.py 中 GPT 的初始化，用注释#remain 标出
外面的 titans.py 和 models.py 均没用，后期删掉即可
测试数据集的生成可以放在 minGPT/test/ 文件夹下
后续的集成训练和测试可以放在 minGPT/文件夹下
minGPT 原始的测试文件等等，在最后可以删掉

现在最需要解决的问题：

1. 对于每一个输入的测试数据(batchsize, prompt)
   minGPT 需要先将每一条 prompt embedding 成 x = (batchsize, token, embedding)
   SY 暂时未找到 minGPT 的 enbedding 逻辑
2. 针对输入的测试数据(batchsize, prompt)
   SY 不清楚 DC 在测试和训练需要的输入分别是什么，需要什么维度
   在 train 阶段需要 update 的参数是什么
   DC 输出的形状是什么
