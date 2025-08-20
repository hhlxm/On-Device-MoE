# Introduction
MoE模型是一种计算友好型专家，适合在端侧设备这种计算能力偏弱的场景下部署。

但由于MoE模型本身占据大部分内存，而端侧设备内存优先，所以目前大部分方法采取offloading范式将模型参数下放到更低级的存储结构。

当前基于Next Layer prediction的预取方法通过预测下一层专家，来将下一层专家预取上来，从而减少数据以来带来的推理延迟

但是由于在端侧IO时间和计算时间很难Overlap住（大概1：3的比例），所以我们从单次IO和IO次数的角度来减少IO的负担

减少IO次数：next token prediction 来将下一个token可能激活的专家提前load上来

减少单次IO：Expert sparsity



# TODO
- IO与计算时间
  - 下一层预测准确，还存在IO和计算的差异问题吗，
  - 云端怎么样
  - 各个不同模型的expert的大小不同，需要测量
- 预测准确性
  - 之前是prefill阶段测试准确性
- 缓存是全局还是每层平均
- Sparsity之后，是不是next layer 也能改进