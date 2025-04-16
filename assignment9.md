# Assignment #9: Huffman, BST & Heap

Updated 1834 GMT+8 Apr 15, 2025

2025 spring, Complied by <mark>高景行 数学科学学院</mark>



> **说明：**
>
> 1. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### LC222.完全二叉树的节点个数

dfs, https://leetcode.cn/problems/count-complete-tree-nodes/

思路：

dfs

代码：

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def countNodes(self, root):
        n = 0
        def dfs(x):
            if not x: return 0
            cnt = 1
            if x.left: cnt += dfs(x.left)
            if x.right: cnt += dfs(x.right)
            return cnt
        return dfs(root)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image.png)



### LC103.二叉树的锯齿形层序遍历

bfs, https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/

思路：

bfs

代码：

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[List[int]]
        """
        if not root: return []
        ans = []
        q = [root]
        height = 0
        while q:
            height += 1
            new = []
            result = []
            for x in q:
                if x.left: new.append(x.left)
                if x.right: new.append(x.right)
                result.append(x.val)
            if height % 2: ans.append(result)
            else: ans.append(result[::-1])
            q = new
        return ans        
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-1.png)



### M04080:Huffman编码树

greedy, http://cs101.openjudge.cn/practice/04080/

思路：

直接做

代码：

```python
import heapq
n = int(input())
q = list(map(int, input().split()))
heapq.heapify(q)
ans = 0
while len(q) > 1:
    x, y = heapq.heappop(q), heapq.heappop(q)
    ans += x + y
    heapq.heappush(q, x + y)
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-2.png)



### M05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/

思路：

题目含义不是实现堆（因为不交换节点），而是在插入节点的时候保持左<中<右

代码：

```python
import heapq
from collections import deque

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def insert(rt, val):
    if not rt: return Node(val)
    if val < rt.val: rt.left = insert(rt.left, val)
    else: rt.right = insert(rt.right, val)
    return rt

a = list(map(int, input().split()))
seen = set()
rt = None
for x in a:
    if x in seen: continue
    seen.add(x)
    rt = insert(rt, x)
q = deque([rt])
result = []
while q:
    x = q.popleft()
    if x.left: q.append(x.left)
    if x.right: q.append(x.right)
    result.append(x.val)
print(" ".join(map(str, result)), end = "")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>


![alt text](image-3.png)


### M04078: 实现堆结构

手搓实现，http://cs101.openjudge.cn/practice/04078/

类似的题目是 晴问9.7: 向下调整构建大顶堆，https://sunnywhy.com/sfbj/9/7

思路：

模版题，学到了新东西。有很多细节。不知道上机考试考不考手搓堆的题目，还是说堆的题目都可以直接调用heapq解决。

代码：

```python
class Heap: # left = node * 2, right = node * 2 + 1 # root是最小值！小根堆
    def __init__(self):
        self.nodes = [-1]
        self.size = 0

    """def build(self, a = []):
        m = len(a) >> 1  # m+1~n都是叶子，不用管
        self.nodes = [-1] + a[:] # 最好切片成新的list
        while m > 0:
            self.percdown(m)
            m -= 1"""

    def percup(self, idx):
        while (idx >> 1) > 0:
            if self.nodes[idx] < self.nodes[idx >> 1]:
                tmp = self.nodes[idx >> 1]
                self.nodes[idx >> 1] = self.nodes[idx]
                self.nodes[idx] = tmp
            idx >>= 1

    def insert(self, val):
        self.nodes.append(val)
        self.size += 1
        self.percup(self.size)

    def min_index(self, idx): # idx子树中最小值，不包括idx自身
        if (idx << 1 | 1) > self.size: return idx << 1
        if self.nodes[idx << 1] < self.nodes[idx << 1 | 1]: return idx << 1
        return idx << 1 | 1

    def percdown(self, idx):
        while (idx << 1) <= self.size:
            min_idx = self.min_index(idx) # 找到最小值位置
            if self.nodes[idx] > self.nodes[min_idx]:
                tmp = self.nodes[idx]
                self.nodes[idx] = self.nodes[min_idx]
                self.nodes[min_idx] = tmp
            idx = min_idx

    def delete(self):
        ans = self.nodes[1]
        self.nodes[1] = self.nodes[self.size]
        self.size -= 1
        self.nodes.pop()
        self.percdown(1)
        return ans

T = int(input())
q = Heap()
for __ in range(T):
    data = input().strip().split()
    if data[0] == '1': q.insert(int(data[1]))
    else: print(q.delete())
```

晴问9.7题目：

```python
def adjust_heap(heap, low, high):
    i = low; j = i << 1
    while j <= high:
        if j + 1 <= high and heap[j + 1] > heap[j]: j += 1
        if heap[j] > heap[i]:
            heap[i], heap[j] = heap[j], heap[i]
            i = j; j = i << 1
        else: break

def create_heap(heap, n):
    for i in range(n >> 1, 0, -1):
        adjust_heap(heap, i, n)

n = int(input())
heap = [0] + list(map(int, input().split()))
create_heap(heap, n)
print(*heap[1:])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-4.png)



### T22161: 哈夫曼编码树

greedy, http://cs101.openjudge.cn/practice/22161/

思路：

学习了huffman编码原理，但根据原理再做就非常简单了，基本没有难的细节。

代码：

```python
import heapq
from curses.ascii import isalpha


class Node:
    def __init__(self, val, weight, leaf):
        self.val = val
        self.leaf = leaf
        self.weight = weight
        self.left = None
        self.right = None

    def __lt__(self, other): # weight越小的项越小
        if self.weight != other.weight: return self.weight < other.weight
        return self.val < other.val

def solve1(data):
    cur = rt
    i = 0
    result = ""
    while i <= len(data) - 1:
        if cur.leaf:
            result += cur.val
            cur = rt
        if data[i] == '0': cur = cur.left
        else: cur = cur.right
        i += 1
    if cur.leaf: result += cur.val
    print(result)

def pre_process(x, string):
    if x.leaf:
        dict[x.val] = string
        return
    if x.left: pre_process(x.left, string + "0")
    if x.right: pre_process(x.right, string + "1")

def solve2(data):
    result = ""
    for x in data: result += dict[x]
    print(result)

n = int(input())
q = []
dict = {}
for i in range(n):
    data = input().split()
    heapq.heappush(q, Node(data[0], int(data[1]), True))
for ___ in range(n - 1):
    x, y = heapq.heappop(q), heapq.heappop(q)
    father = Node(min(x.val, y.val), x.weight + y.weight, False)
    father.left, father.right = x, y
    heapq.heappush(q, father)

rt = q[0]
pre_process(rt, "")
while True:
    try:
        data = input()
        if data.isdigit(): solve1(data)
        else: solve2(data)
    except EOFError: break
```

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-5.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

本次作业有难度，几个手搓堆的题目学到了堆的内在算法和建二叉树解决的题目（比如huffman）。本周还是很忙，下周应该就能追赶每日选做。









