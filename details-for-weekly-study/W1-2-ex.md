以下是针对**第1-2周学习内容**的测试题，分为**Python编程/数据处理**和**数学基础/算法**两大模块，包含选择题、编程题和简答题。完成后可对照答案自测掌握程度。

---

### **模块1：Python编程与数据处理**
#### **题目1：Python基础语法**  
**代码题**：  
编写一个函数 `fibonacci(n)`，返回斐波那契数列的前`n`项（用列表存储）。  
示例：`fibonacci(5) → [0, 1, 1, 2, 3]`

---

#### **题目2：面向对象编程**  
**代码题**：  
定义一个`Student`类，包含属性`name`（字符串）和`grades`（列表），并实现以下方法：  
- `add_grade(score)`: 向`grades`中添加一个分数（0-100之间的整数）  
- `average_score()`: 返回平均分（保留两位小数）  

示例：  
```python  
s = Student("Alice")  
s.add_grade(80)  
s.add_grade(90)  
print(s.average_score())  # 输出 85.00  
```

---

#### **题目3：Pandas数据处理**  
**场景题**：  
给定Titanic数据集（包含`Survived`、`Pclass`、`Age`等列），请用Pandas完成以下操作：  
1. 计算每个舱位（`Pclass`）乘客的平均年龄。  
2. 找出年龄大于30岁且幸存（`Survived=1`）的乘客数量。  
请写出完整代码（假设数据集已加载为`df`）。

---

#### **题目4：数据可视化**  
**场景题**：  
使用Matplotlib绘制一条曲线图，横轴为`x = np.linspace(0, 2*np.pi, 100)`，纵轴为`y = np.sin(x)`，要求：  
- 设置标题为“Sine Wave”  
- 横纵轴标签分别为“Angle (rad)”和“sin(x)”  
- 保存为PNG文件`sine.png`  

请写出完整代码。

---

### **模块2：数学基础与算法**  
#### **题目5：线性代数**  
**计算题**：  
给定矩阵：
$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad$ 
$B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$
计算 $A \times B$（矩阵乘法）和 $A \circ B$（哈达玛积，即逐元素相乘）的结果。

---

#### **题目6：微积分**  
**简答题**：  
设函数 $f(x, y) = x^3 + 2xy + y^2$，求：  
1. 偏导数 $\frac{\partial f}{\partial x}$ 和 $\frac{\partial f}{\partial y}$  
2. 在点 $(1, 2)$ 处的梯度向量 $\nabla f$

---

#### **题目7：概率与统计**  
**计算题**：  
假设某疾病检测的准确率为：  
- 患病者检测为阳性的概率为 95%（灵敏度）  
- 健康人检测为阴性的概率为 90%（特异度）  
已知该疾病在人群中的患病率为 1%。  
问题：若某人检测为阳性，实际患病的概率是多少？（用贝叶斯定理计算）

---

#### **题目8：数据结构与算法**  
**代码题**：  
实现快速排序算法（升序），函数签名为 `def quick_sort(arr: list) -> list`。  
示例：`quick_sort([3, 1, 4, 1, 5]) → [1, 1, 3, 4, 5]`

---

#### **题目9：LeetCode风格题**  
**编程题**（任选其一）：  
1. [Two Sum](https://leetcode.com/problems/two-sum/)：找出数组中两数之和等于目标值的下标。  
2. [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)：反转单链表。

---

### **参考答案与解析**  
#### **模块1答案**  
**题目1**：  
```python  
def fibonacci(n):  
    if n == 0:  
        return []  
    elif n == 1:  
        return [0]  
    res = [0, 1]  
    for _ in range(n-2):  
        res.append(res[-1] + res[-2])  
    return res[:n]  
```  

**题目2**：  
```python  
class Student:  
    def __init__(self, name):  
        self.name = name  
        self.grades = []  

    def add_grade(self, score):  
        if 0 <= score <= 100:  
            self.grades.append(score)  

    def average_score(self):  
        return round(sum(self.grades)/len(self.grades), 2) if self.grades else 0.0  
```  

**题目3**：  
```python  
# 问题1  
print(df.groupby('Pclass')['Age'].mean())  

# 问题2  
condition = (df['Age'] > 30) & (df['Survived'] == 1)  
print(len(df[condition]))  
```  

**题目4**：  
```python  
import matplotlib.pyplot as plt  
import numpy as np  

x = np.linspace(0, 2*np.pi, 100)  
y = np.sin(x)  

plt.plot(x, y)  
plt.title("Sine Wave")  
plt.xlabel("Angle (rad)")  
plt.ylabel("sin(x)")  
plt.savefig("sine.png")  
plt.close()  
```  

---

#### **模块2答案**
**题目5**：  
- **矩阵乘法**：  
  $$
  \begin{aligned}
  A \times B &= \begin{bmatrix} 
  1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 
  3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 
  \end{bmatrix} \\
  &= \begin{bmatrix} 
  19 & 22 \\ 
  43 & 50 
  \end{bmatrix}
  \end{aligned}
  $$  

- **哈达玛积**：  
  $$
  A \circ B = \begin{bmatrix} 
  1 \cdot 5 & 2 \cdot 6 \\ 
  3 \cdot 7 & 4 \cdot 8 
  \end{bmatrix} = \begin{bmatrix} 
  5 & 12 \\ 
  21 & 32 
  \end{bmatrix}
  $$

---

**题目6**：  
1. **偏导数**：  
   $$
   \frac{\partial f}{\partial x} = 3x^2 + 2y, \quad 
   \frac{\partial f}{\partial y} = 2x + 2y
   $$  

2. **梯度向量**：  
   $$
   \nabla f(1, 2) = \left( 3(1)^2 + 2(2),\ 2(1) + 2(2) \right) = (7, 6)
   $$

---

**题目7**：  
**贝叶斯定理计算过程**：  
$$
\begin{aligned}
P(\text{患病} | \text{阳性}) 
&= \frac{P(\text{阳性}|\text{患病}) P(\text{患病})}{P(\text{阳性})} \\
P(\text{阳性}) 
&= P(\text{阳性}|\text{患病}) P(\text{患病}) + P(\text{阳性}|\text{健康}) P(\text{健康}) \\
&= (0.95 \times 0.01) + (0.10 \times 0.99) \\
&= 0.0095 + 0.099 \\
&= 0.1085 \\
\Rightarrow P(\text{患病} | \text{阳性}) 
&= \frac{0.0095}{0.1085} \approx 8.76\%
\end{aligned}
$$ 

**题目8**：  
```python  
def quick_sort(arr):  
    if len(arr) <= 1:  
        return arr  
    pivot = arr[len(arr)//2]  
    left = [x for x in arr if x < pivot]  
    middle = [x for x in arr if x == pivot]  
    right = [x for x in arr if x > pivot]  
    return quick_sort(left) + middle + quick_sort(right)  
```  

**题目9**（以Two Sum为例）：  
```python  
def two_sum(nums, target):  
    hashmap = {}  
    for i, num in enumerate(nums):  
        complement = target - num  
        if complement in hashmap:  
            return [hashmap[complement], i]  
        hashmap[num] = i  
    return []  
```  

---

### **自测建议**  
1. 独立完成所有题目后再看答案。  
2. **评分标准**：  
   - 编程题：代码逻辑正确且能运行 → 满分  
   - 数学题：公式推导正确 → 满分  
   - 正确率 ≥ 80%：掌握良好，可进入下一阶段  
   - 正确率 ≤ 60%：需复习薄弱环节（如重看3Blue1Brown视频或重写代码）  
3. **延伸练习**：  
   - 如果贝叶斯定理不熟，尝试用Python模拟计算条件概率。  
   - 如果快速排序不熟练，手动模拟排序过程（如对[5,3,1]排序）。  

通过完成这些题目，你可以清晰定位自己的知识盲点。如果有具体题目需要进一步解析，请随时告诉我！