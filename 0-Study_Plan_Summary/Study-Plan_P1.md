# 深度学习实习生12周学习规划

以下是一份为期12周的详细学习规划，覆盖代码能力、机器学习/深度学习理论与实践，时间安排精确到每周和每天，帮助你系统性提升能力。根据你的基础可适当调整节奏。

---

## **总体目标**  
1. **代码能力**：熟练掌握Python、数据结构与算法、PyTorch框架  
2. **机器学习基础**：掌握经典算法（线性回归、SVM、决策树等）  
3. **深度学习核心**：掌握CNN/RNN/Transformer，完成2-3个完整项目  
4. **求职准备**：LeetCode刷题、面试问题模拟、简历优化

---

## **时间规划表（12周）**

### **第1-2周：Python编程与数学基础**
**目标**：巩固Python编程能力，复习核心数学知识  
**每日安排**（每周6天，每天4-5小时）：  
- **上午（2小时）**  
  - *Python语法（第1周）*：  
    - 学习基础语法、函数、面向对象编程（推荐《Python Crash Course》）  
    - 刷题练习：[HackerRank Python模块](https://www.hackerrank.com/domains/python)（每天5题）  
  - *数学基础（第2周）*：  
    - 线性代数：矩阵乘法、特征值（[3Blue1Brown视频](https://www.bilibili.com/video/BV1ys411472E)）  
    - 微积分：梯度、链式法则（《深度学习》花书第2章）  
- **下午（2小时）**  
  - *Numpy/Pandas实战*：  
    - 完成Kaggle的[Python微课程](https://www.kaggle.com/learn/python)（Pandas、Data Visualization）  
    - 用Numpy实现线性回归、矩阵分解  
- **晚上（1小时）**  
  - **LeetCode**：每日1-2道简单题（[数组](https://leetcode.com/tag/array/)、[字符串](https://leetcode.com/tag/string/)）  

---

### **第3-4周：数据结构与算法**
**目标**：掌握常见算法，应对技术面试  
**每日安排**：  
- **上午（2小时）**  
  - *算法学习*：  
    - 排序（快速排序、归并排序）、搜索（二分查找）、动态规划（背包问题）  
    - 参考书籍：《算法导论》或《算法图解》  
- **下午（3小时）**  
  - **LeetCode刷题**：  
    - 按类型刷题（每天3-4题）：[数组](https://leetcode.com/tag/array/)、[链表](https://leetcode.com/tag/linked-list/)、[二叉树](https://leetcode.com/tag/binary-tree/)（第3周）  
    - [动态规划](https://leetcode.com/tag/dynamic-programming/)、[回溯](https://leetcode.com/tag/backtracking/)（第4周）  
    - 重点：[Top 100高频题](https://leetcode.com/problem-list/79h8rn6/)  
- **晚上（1小时）**  
  - *复盘*：整理错题本，总结代码模板  

---

### **第5-6周：机器学习基础**
**目标**：掌握经典机器学习算法及Scikit-learn实现  
**每日安排**：  
- **上午（2小时）**  
  - *理论学习*：  
    - 监督学习：线性回归、逻辑回归、SVM、决策树（参考《Hands-On ML》）  
    - 无监督学习：K-Means、PCA  
- **下午（3小时）**  
  - *项目实战*：  
    - 使用Scikit-learn完成Kaggle项目：  
      - [Titanic生存预测](https://www.kaggle.com/c/titanic)  
      - [房价回归](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
    - 掌握交叉验证、特征工程、模型评估（AUC、F1 Score）  
- **晚上（1小时）**  
  - *论文阅读*：精读经典论文《[Random Forests](https://link.springer.com/article/10.1023/A:1010933404324)》  

---

### **第7-8周：深度学习入门（PyTorch）**
**目标**：掌握PyTorch框架，实现基础神经网络  
**每日安排**：  
- **上午（2小时）**  
  - *PyTorch语法*：  
    - 张量操作、自动求导、Dataset/Dataloader  
    - 从零编写全连接网络（MNIST分类）  
- **下午（3小时）**  
  - *项目实战*：  
    - 复现经典模型：[LeNet-5](http://yann.lecun.com/exdb/lenet/)、[AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)（CIFAR-10数据集）  
    - 学习调参技巧：学习率调整、早停法（Early Stopping）  
- **晚上（1小时）**  
  - *文档阅读*：[PyTorch官方教程](https://pytorch.org/tutorials/)  

---

### **第9-10周：深度学习进阶（CV/NLP方向）**
**目标**：专攻一个领域（如CV），掌握前沿模型  
**每日安排**（以计算机视觉为例）：  
- **上午（2小时）**  
  - *模型学习*：  
    - CNN进阶：[ResNet](https://arxiv.org/abs/1512.03385)、[YOLO](https://arxiv.org/abs/1506.02640)（第9周）  
    - Transformer：[ViT](https://arxiv.org/abs/2010.11929)、[Swin Transformer](https://arxiv.org/abs/2103.14030)（第10周）  
- **下午（3小时）**  
  - *项目实战*：  
    - 使用预训练模型（如ResNet-50）进行迁移学习（自定义数据集）  
    - 实现图像分割任务：[U-Net](https://arxiv.org/abs/1505.04597) + [PASCAL VOC数据集](http://host.robots.ox.ac.uk/pascal/VOC/)  
- **晚上（1小时）**  
  - *论文复现*：精读并复现一篇[CVPR论文](https://cvpr2023.thecvf.com/)的核心部分  

---

### **第11-12周：项目整合与求职准备**
**目标**：完成端到端项目，准备面试  
**每日安排**：  
- **上午（2小时）**  
  - *高级项目*：  
    - 全流程项目：数据清洗→模型训练→部署（如用[Flask](https://flask.palletsprojects.com/)部署图像分类API）  
    - 优化模型性能：模型量化（Quantization）、剪枝（Pruning）  
- **下午（2小时）**  
  - *面试准备*：  
    - 高频面试题：过拟合解决方法、梯度消失/爆炸、Attention机制  
    - 行为问题：用[STAR法则](https://www.themuse.com/advice/star-interview-method)描述项目难点与解决方案  
- **晚上（2小时）**  
  - *简历与复盘*：  
    - 撰写简历：突出技术栈（如PyTorch、CNN）和项目指标（如准确率提升5%）  
    - 模拟面试：使用[Interviewing.io](https://interviewing.io/)模拟技术面  

---

## **关键执行建议**  
1. **每日复盘**：睡前30分钟整理当日学习内容（Notion或纸质笔记）  
2. **代码托管**：所有项目代码上传GitHub，形成作品集  
3. **社区参与**：每周参与1次[Kaggle讨论](https://www.kaggle.com/discussion)或[Stack Overflow答疑](https://stackoverflow.com/)  
4. **弹性时间**：每周留1天休息或补进度，避免过度疲劳  

---

通过坚持这一规划，你将在3个月内构建扎实的代码能力与深度学习知识体系，并为实习申请做好充分准备。