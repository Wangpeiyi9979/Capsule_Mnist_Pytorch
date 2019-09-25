# 项目环境
- Pytorch: 1.x
- Python3.6

# 项目简介
本项目使用`Pytorch`复现了简单的`CNN+Capsule`网络。
实验为【单数字训练，双数字测试】，实验结果证明了Capsule网络【举一反三】的特性，本项目是[该项目](https://kexue.fm/archives/4819)的`Pytorch`复现

# 使用
在主目录下执行以下命令
- 使用CNN模型

    ```
    python main.py train --model="CNN"
    ```
- 使用CNN+Capsule
    ```
    python main.py train --model="CNN+Capsule"
    ```
# 结果
最后【任务一：单数字训练，单数字测试】任务结果两个模型相差无几，但是【任务二：单数字训练，双数字测试】任务两者差别巨大

模型|任务一(ACC) | 任务二(ACC)
---|---|---
CNN|98.7\% | 22.0\%
CNN+Capsule|99.0\% | 94.0\%

# 详细代码解读
https://wangpeiyi.blog.csdn.net/article/details/101352988

