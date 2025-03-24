# Assignment #6: 回溯、树、双向链表和哈希表

Updated 1526 GMT+8 Mar 22, 2025

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

### LC46.全排列

backtracking, https://leetcode.cn/problems/permutations/

思路：

代码中注意要切片加入答案：切片后是新的数组，如果不切片则是原来数组的引用，会随着nums的修改而修改。


代码：

```python
class Solution(object):
    def permute(self, nums):
        def next_permutation(b):
            pos1 = n - 2
            while pos1 >= 0 and b[pos1] >= b[pos1 + 1]: pos1 -= 1
            if pos1 == -1: return [i + 1 for i in range(n)]
            pos2 = n - 1
            while pos2 >= 0 and b[pos2] <= b[pos1]: pos2 -= 1
            b[pos2], b[pos1] = b[pos1], b[pos2]
            b[pos1 + 1:] = sorted(b[pos1 + 1:])
            return b
        nums, n, T = sorted(nums), len(nums), 1
        for i in range(1, n + 1): T *= i
        result = []
        for i in range(T):
            result.append(nums[:]) # ！
            nums = next_permutation(nums)
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](image.png)




### LC79: 单词搜索

backtracking, https://leetcode.cn/problems/word-search/

思路：

直接做


代码：

```python
class Solution(object):
    def exist(self, board, word):
        self.FLAG = False
        def dfs(x, y, k):
            if k == len(word):
                self.FLAG = True
                return
            if self.FLAG: return
            for nx, ny in directions:
                tx, ty = x + nx, y + ny
                if 0 <= tx < n and 0 <= ty < m and not flag[tx][ty] and board[tx][ty] == word[k]:
                    flag[tx][ty] = True
                    dfs(tx, ty, k + 1)
                    flag[tx][ty] = False
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        n, m = len(board), len(board[0])
        flag = [[False] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if board[i][j] == word[0]:
                    flag[i][j] = True
                    dfs(i, j, 1)
                    flag[i][j] = False
                    if self.FLAG: break
            if self.FLAG: break
        return self.FLAG
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](image-1.png)




### LC94.二叉树的中序遍历

dfs, https://leetcode.cn/problems/binary-tree-inorder-traversal/

思路：

直接做


代码：

```python
class Solution(object):
    def inorderTraversal(self, root):
        result = []
        def dfs(x):
            if not x: return
            dfs(x.left)
            result.append(x.val)
            dfs(x.right)
        dfs(root)
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-2.png)



### LC102.二叉树的层序遍历

bfs, https://leetcode.cn/problems/binary-tree-level-order-traversal/

思路：

直接做


代码：

```python
class Solution(object):
    def levelOrder(self, root):
        q = deque([(root, 0)])
        result, current_depth = [], -1
        while q:
            x, d = q.popleft()
            if not x: continue
            if d != current_depth:
                current_depth = d
                result.append([x.val])
            else: result[-1].append(x.val)
            q.append([x.left, d + 1]); q.append([x.right, d + 1])
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-3.png)



### LC131.分割回文串

dp, backtracking, https://leetcode.cn/problems/palindrome-partitioning/

思路：

直接做即可。后面发现递归稍快一点（感觉是常数上的优化而非本质优化）


代码：

```python
class Solution(object):
    def partition(self, s):
        n = len(s)
        possible = [[] for i in range(n)]
        dp = [[True] * n for i in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                dp[i][j] = dp[i + 1][j - 1] and (s[i] == s[j])
        for j in range(n):
            for i in range(j + 1):
                if dp[i][j]: possible[j].append(i)
        split_result = [[] for i in range(n + 1)]; split_result[-1].append([-1])
        for i in range(n):
            for x in possible[i]:
                if split_result[x - 1]:
                    for t in split_result[x - 1]:
                        split_result[i].append(t + [i])
        ans = []
        for x in split_result[n - 1]:
            temporary = []
            for i in range(len(x) - 1):
                temporary.append(s[x[i] + 1: x[i + 1] + 1])
            ans.append(temporary)
        return ans
```

```python
class Solution(object):
    def partition(self, s):

        def check(x, y):
            return s[x:y + 1] == s[x:y + 1][::-1]
        n = len(s)
        dp = [[True] * n for i in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                dp[i][j] = dp[i + 1][j - 1] and (s[i] == s[j])
        result, ans = list(), list()
        def dfs(x):
            if x == n:
                result.append(ans[:])
                return
            for j in range(x, n):
                if dp[x][j]:
                    ans.append(s[x:j + 1])
                    dfs(j + 1)
                    ans.pop()
        dfs(0)
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-4.png)



### LC146.LRU缓存

hash table, doubly-linked list, https://leetcode.cn/problems/lru-cache/

思路：

这题直接做就挺简单的。后来问了AI发现deque的in操作竟然是O(len)的，所以刚开始的做法比较差。

于是下面改成了用counting而不用deque的in操作的做法，立马变成O(1)飞快。

后来看了题解学了更快的双向链表的做法。（因为本题相当于用链表实现一个有更强功能的deque）


代码：

```python
class LRUCache(object):
    def __init__(self, capacity):
        self.remains = capacity
        self.lru = {}
        self.in_lru = [False] * 10001
        self.time_stamp = deque()

    def get(self, key):
        if not self.in_lru[key]: return -1
        self.time_stamp.append(key)
        return self.lru[key]
    
    def put(self, key, value):
        if self.in_lru[key]: 
            self.lru[key] = value
            self.time_stamp.append(key)
            return
        self.lru[key] = value
        self.in_lru[key] = True
        self.time_stamp.append(key)
        self.remains -= 1
        if self.remains == -1:
            while self.time_stamp:
                key = self.time_stamp.popleft()
                if key in self.time_stamp: continue
                self.in_lru[key] = False
                self.remains = 0
                break
```

```python
class LRUCache(object):
    def __init__(self, capacity):
        self.remains = capacity
        self.lru = {}
        self.time_stamp = deque()
        self.counting = [0] * 10001

    def get(self, key):
        if not self.counting[key]: return -1
        self.time_stamp.append(key)
        self.counting[key] += 1
        return self.lru[key]
    
    def put(self, key, value):
        if self.counting[key]: 
            self.lru[key] = value
            self.time_stamp.append(key)
            self.counting[key] += 1
            return
        self.lru[key] = value
        self.time_stamp.append(key)
        self.counting[key] += 1
        self.remains -= 1
        if self.remains == -1:
            while self.time_stamp:
                key = self.time_stamp.popleft()
                self.counting[key] -= 1
                if self.counting[key] >= 1: continue
                self.remains = 0
                break
```

双链表的做法用空的头指针和尾指针，方便快速插入到头部和移除尾部。

dict的pop(key)时间复杂度是O(1)！！

```python
class Node:
    def __init__(self, key = -1, value = -1):
        self.key = key
        self.value = value
        self.next = None
        self.pre = None

class LRUCache(object):
    def __init__(self, capacity):
        self.remain = capacity
        self.lru = {} # key -> Node
        self.head = Node()
        self.tail = Node() # 伪头部和伪尾部
        self.head.next, self.tail.pre = self.tail, self.head

    def get(self, key):
        if key not in self.lru: return -1
        new_node = self.lru[key]
        self.move_to_head(new_node)
        return new_node.value
    
    def put(self, key, value):
        if key not in self.lru:
            new_node = Node(key, value)
            self.lru[key] = new_node
            self.add_to_head(new_node)
            self.remain -= 1
            if self.remain == -1:
                removed = self.remove_tail()
                self.lru.pop(removed.key) # dict的pop(key)时间复杂度是O(1)!!!
                self.remain = 0
        else:
            new_node = self.lru[key]
            new_node.value = value
            self.move_to_head(new_node)

    def add_to_head(self, node):
        node.pre = self.head
        node.next = self.head.next
        self.head.next.pre = node
        self.head.next = node

    def move_to_head(self, node):
        node.pre.next = node.next
        node.next.pre = node.pre
        self.add_to_head(node)

    def remove_tail(self):
        node = self.tail.pre
        node.pre.next = node.next
        node.next.pre = node.pre
        return node
```

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-6.png)

![alt text](image-5.png)


## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

本周题目有难度，通过题解学到了很多链表的应用，加深了对链表结构的理解。

由于这周实在太忙没跟上每日选做，下周争取补完。







