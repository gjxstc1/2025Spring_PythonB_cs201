# Assignment #8: 树为主

Updated 1704 GMT+8 Apr 8, 2025

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

### LC108.将有序数组转换为二叉树

dfs, https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

思路：

直接做


代码：

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sortedArrayToBST(self, nums):
        def dfs(l, r):
            if l > r: return None
            if l == r: return TreeNode(nums[l])
            mid = (l + r) >> 1
            node = TreeNode(nums[mid])
            node.left = dfs(l, mid - 1)
            node.right = dfs(mid + 1, r)
            return node
        return dfs(0, len(nums) - 1)
        
        
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image.png)



### M27928:遍历树

 adjacency list, dfs, http://cs101.openjudge.cn/practice/27928/

思路：

输入处理起来比较麻烦；最后参考了题解的优雅做法。

代码：

```python
from collections import defaultdict
n = int(input())
tree = defaultdict(list)
parent, children = set(), set()
for i in range(n):
    x = list(map(int, input().split()))
    parent.add(x[0])
    if len(x) > 1:
        tmp = x[1:]
        children.update(tmp)
        tree[x[0]] = tmp

def traversal(node):
    order = sorted(tree[node] + [node])
    for x in order:
        if x != node: traversal(x)
        else: print(x)

rt = (parent - children).pop()
traversal(rt)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-1.png)



### LC129.求根节点到叶节点数字之和

dfs, https://leetcode.cn/problems/sum-root-to-leaf-numbers/

思路：

nonlocal关键字竟然在python中编译不通过，只能选择python3（不知道考试环境是python还是python3）。


代码：

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumNumbers(self, root):
        ans = 0
        def dfs(node, cur):
            nonlocal ans
            cur = cur * 10 + node.val
            if not node.left and not node.right: 
                ans += cur
                return 
            if node.left: dfs(node.left, cur)
            if node.right: dfs(node.right, cur)
        dfs(root, 0)
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-2.png)



### M22158:根据二叉树前中序序列建树

tree, http://cs101.openjudge.cn/practice/22158/

思路：

直接做。经典题目

代码：

```python
from collections import defaultdict
class Tree:
    def __init__(self, val, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right


def dfs(pre_l, pre_r, in_l, in_r):
    if pre_l > pre_r or in_l > in_r: return None
    if pre_l == pre_r: return Tree(pre_order[pre_l])
    index = in_order.index(pre_order[pre_l])
    length = index - in_l
    new_node = Tree(pre_order[pre_l])
    new_node.left = dfs(pre_l + 1, pre_l + length, in_l, index - 1)
    new_node.right = dfs(pre_l + length + 1, pre_r, index + 1, in_r)
    return new_node

def post_order(x):
    if x.left: post_order(x.left)
    if x.right: post_order(x.right)
    print(x.val, end = "")

while True:
    try:
        pre_order = input()
        in_order = input()
        tree = defaultdict(str)
        n = len(pre_order)
        rt = dfs(0, n - 1, 0, n - 1)
        post_order(rt)
        print()
    except EOFError: break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-3.png)



### T24729:括号嵌套树

dfs, stack, http://cs101.openjudge.cn/practice/24729/

思路：

直接做。感觉本题应该不到tough难度。注意特判只有一个点的情况。

代码：

```python
class Tree:
    def __init__(self, val):
        self.val = val
        self.son = []

def pre_order(x):
    print(x.val, end = "")
    for y in x.son: pre_order(y)

def post_order(x):
    for y in x.son: post_order(y)
    print(x.val, end = "")

s = input()
q = []
if len(s) == 1:
    print(s)
    print(s)
    exit(0)
cur_node = None
for x in s[:-1]:
    if x == '(':
        q.append(cur_node)
    elif x == ')': q.pop()
    elif x == ',': continue
    else:
        cur_node = Tree(x)
        if q: q[-1].son.append(cur_node)

rt = q[-1]
pre_order(rt)
print()
post_order(rt)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-4.png)



### LC3510.移除最小数对使数组有序II

doubly-linked list + heap, https://leetcode.cn/problems/minimum-pair-removal-to-sort-array-ii/

思路：

思路容易但是细节非常繁琐，比较困难。用链表表示存活的点之间的连接关系，关键是用懒标记来计数处理所有在删除数后应该修改的内容（相邻两数的和）。

代码：

```python
class Solution(object):
    def minimumPairRemoval(self, nums):
        n = len(nums)
        prev = [i - 1 for i in range(n + 1)]
        next = [i + 1 for i in range(n + 1)]
        tail = n - 1 # head = 0 保持不动
        q = []
        cnt = 0 # nums[i] > nums[next[i]]的个数
        for i in range(n - 1): heapq.heappush(q, (nums[i] + nums[next[i]], i))
        for i in range(n - 1):
            if nums[i] > nums[i + 1]: cnt += 1
        lazy = defaultdict(int)
        epoch = 0
        while q and cnt:
            epoch += 1
            while lazy[q[0]]:
                lazy[q[0]] -= 1
                heapq.heappop(q)
            
            val, pos = heapq.heappop(q) # val = nums[pos] + nums[next[pos]]
            x, y = pos, next[pos]

            if nums[x] > nums[y]: cnt -= 1
            prex, posy = prev[x], next[y]
            if prex >= 0: 
                if nums[prex] > nums[x]: cnt -= 1
                if nums[prex] > val: cnt += 1
                lazy[(nums[prex] + nums[x], prex)] += 1
                heapq.heappush(q, (nums[prex] + val, prex))
            if y < tail:
                if nums[y] > nums[posy]: cnt -= 1
                if val > nums[posy]: cnt += 1
                lazy[(nums[y] + nums[posy], y)] += 1
                heapq.heappush(q, (val + nums[posy], x))
            else: tail = x

            nums[x] = val # 这行一定要写!!!!

            next[x] = posy
            prev[posy] = x

        return epoch
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-5.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

本周超级忙，因此没有抽出时间练习。并且感觉本次作业并不简单。期中考试结束后再赶上进度。









