以下是针对**第3-4周（数据结构与算法）**的测试题，包含编程题、理论题和数学分析题，并附带详细答案与解析：

---

### **模块1：数据结构与基础算法**
#### **题目1：链表操作**  
**编程题**：  
实现一个函数 `reverse_between(head, m, n)`，反转单链表从第`m`个节点到第`n`个节点的部分（位置从1开始计数）。  
示例：  
输入：`1->2->3->4->5->NULL`, m=2, n=4  
输出：`1->4->3->2->5->NULL`

---

#### **题目2：二叉树遍历**  
**场景题**：  
给定二叉树的前序遍历 `preorder = [3,9,20,15,7]` 和中序遍历 `inorder = [9,3,15,20,7]`，要求：  
1. 重建二叉树  
2. 实现分层遍历（BFS）并返回结果  
示例输出：`[[3], [9,20], [15,7]]`

---

#### **题目3：堆应用**  
**编程题**：  
设计一个实时获取数据流中第K大元素的类（数据流不断添加新元素）。  
类定义：  
```python  
class KthLargest:  
    def __init__(self, k: int, nums: List[int])  
    def add(self, val: int) -> int  
```  
要求：`add` 操作时间复杂度为 O(log K)

---

### **模块2：动态规划与回溯**
#### **题目4：动态规划**  
**理论题**：  
给定一个包含非负整数的网格，找出一条从左上角到右下角的路径，使得路径上的数字总和最小。  
说明：每次只能向下或向右移动一步。  
请：  
1. 写出状态转移方程  
2. 分析空间优化方法  

---

#### **题目5：回溯算法**  
**编程题**：  
生成所有可能的有效括号组合（n对括号）。  
示例：  
输入：n=3  
输出：["((()))","(()())","(())()","()(())","()()()"]

---

#### **题目6：数学分析**  
**复杂度计算**：  
分析快速排序算法的时间复杂度：  
1. 最优情况  
2. 最坏情况  
3. 平均情况  
要求用主定理（Master Theorem）或递推公式证明。

---

### **参考答案与解析**

#### **题目1答案**  
```python  
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_between(head, m, n):
    dummy = ListNode(0)
    dummy.next = head
    pre = dummy
    
    # 移动到m-1位置
    for _ in range(m-1):
        pre = pre.next
    
    # 反转m到n节点
    curr = pre.next
    prev = None
    for _ in range(n - m + 1):
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    
    # 连接头尾
    pre.next.next = curr
    pre.next = prev
    
    return dummy.next
```  
**解析**：  
- 使用虚拟头节点处理边界条件  
- 三指针法反转子链表（时间复杂度O(n)，空间复杂度O(1)）  
- 关键点：反转后需正确连接前后部分（`pre.next.next = curr`）

---

#### **题目2答案**  
```python  
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(preorder, inorder):
    if not preorder:
        return None
    root_val = preorder[0]
    root = TreeNode(root_val)
    idx = inorder.index(root_val)
    
    root.left = build_tree(preorder[1:idx+1], inorder[:idx])
    root.right = build_tree(preorder[idx+1:], inorder[idx+1:])
    return root

def level_order(root):
    res = []
    queue = [root] if root else []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        res.append(level)
    return res
```  
**解析**：  
- 前序确定根节点，中序划分左右子树（时间复杂度O(n^2)，可用哈希表优化到O(n)）  
- BFS使用队列实现分层遍历（时间复杂度O(n)）

---

#### **题目3答案**  
```python  
import heapq

class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.heap = []
        for num in nums:
            self.add(num)

    def add(self, val):
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0] if len(self.heap) >= self.k else -1
```  
**解析**：  
- 维护大小为K的最小堆（堆顶即第K大元素）  
- 每次插入后保持堆大小≤K（时间复杂度O(n log K)）

---

#### **题目4答案**  
**状态转移方程**：  
```
dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
```  
**空间优化**：  
- 使用一维数组滚动更新：  
  ```python
  dp[j] = grid[i][j] + min(dp[j], dp[j-1])
  ```  
**解析**：  
- 原空间复杂度O(mn)，优化后O(n)  
- 注意处理第一行和第一列的边界条件

---

#### **题目5答案**  
```python  
def generate_parenthesis(n):
    res = []
    def backtrack(s, left, right):
        if len(s) == 2*n:
            res.append(s)
            return
        if left < n:
            backtrack(s+'(', left+1, right)
        if right < left:
            backtrack(s+')', left, right+1)
    backtrack('', 0, 0)
    return res
```  
**解析**：  
- 回溯剪枝条件：左括号数≥右括号数  
- 时间复杂度O(4^n/√n)（卡特兰数）

---

#### **题目6答案**  
**复杂度分析**：  
1. **最优情况**（每次划分均衡）：  
   $$T(n) = 2T(n/2) + O(n) \Rightarrow O(n \log n)$$  
2. **最坏情况**（每次划分极端不平衡）：  
   $$T(n) = T(n-1) + O(n) \Rightarrow O(n^2)$$  
3. **平均情况**：  
   递推式：  
   $$T(n) = \frac{2}{n}\sum_{k=0}^{n-1}T(k) + O(n)$$  
   数学证明可得：$$T(n) = O(n \log n)$$  

---

### **自测评分标准**
- **6题全对**：已掌握核心算法，可进入后续学习  
- **正确4-5题**：需复习动态规划状态转移或回溯剪枝策略  
- **正确≤3题**：建议重新学习链表和二叉树模块  

通过这些问题，你可以清晰定位算法实现的薄弱环节。如果需要某题的更详细解析，请随时告知！
