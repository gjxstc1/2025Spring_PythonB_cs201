# Assignment #A: Graph starts

Updated 1830 GMT+8 Apr 22, 2025

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

### M19943:图的拉普拉斯矩阵

OOP, implementation, http://cs101.openjudge.cn/practice/19943/

要求创建Graph, Vertex两个类，建图实现。

思路：

本题本身很容易，这里用类实现了，感觉有点麻烦。

代码：

```python
class Vertex:
    def __init__(self, key):
        self.key = key
        self.neighbour = {}

    def add_edge(self, to, weight):
        self.neighbour[to] = weight

class Graph:
    def __init__(self):
        self.vertices = {}

    def add_vertex(self, key):
        new_vertex = Vertex(key)
        self.vertices[key] = new_vertex

    def add_edge(self, u, v, w = 0):
        if u not in self.vertices: self.add_vertex(u)
        if v not in self.vertices: self.add_vertex(v)
        self.vertices[u].add_edge(self.vertices[v], w)

graph = Graph()
n, m = map(int, input().split())
for i in range(n): graph.add_vertex(i)
edges = []
for _ in range(m):
    x, y = map(int, input().split())
    graph.add_edge(x, y)
    graph.add_edge(y, x)

for vertex in graph.vertices.values():
    row = [0] * n
    row[vertex.key] = len(vertex.neighbour)
    for to in vertex.neighbour.keys(): row[to.key] = -1
    print(' '.join(map(str, row)))

```

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image.png)



### LC78.子集

backtracking, https://leetcode.cn/problems/subsets/

思路：

dfs

代码：

```python
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n = len(nums)
        result = []
        ans = []
        def dfs(x):
            if x == n:
                result.append(ans[:])
                return
            dfs(x + 1)
            ans.append(nums[x])
            dfs(x + 1)
            ans.pop()
        dfs(0)
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-1.png)



### LC17.电话号码的字母组合

hash table, backtracking, https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

思路：

dfs

代码：

```python
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        dict = {"2": list("abc"), "3": list("def"), "4": list("ghi"), "5": list("jkl"), "6": list("mno"), "7": list("pqrs"), "8": list("tuv"), "9": list("wxyz")}
        result = []
        n = len(digits)
        if not n: return []
        def dfs(x, s):
            if x == n:
                result.append(s)
                return 
            for c in dict[digits[x]]: dfs(x + 1, s + c)
        dfs(0, "")
        return result
        
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-2.png)



### M04089:电话号码

trie, http://cs101.openjudge.cn/practice/04089/

思路：

学习trie树的原理。

代码：

```python
class Node:
    def __init__(self, key = None):
        self.son = {}
        self.key = key

class Trie:
    def __init__(self):
        self.rt = Node()

    def insert(self, nums):
        cur = self.rt
        for x in nums:
            if x not in cur.son.keys(): cur.son[x] = Node(x)
            cur = cur.son[x]

    def find(self, nums):
        cur = self.rt
        for x in nums:
            if x not in cur.son.keys(): return False
            cur = cur.son[x]
        return True

T = int(input())
for __ in range(T):
    tree = Trie()
    m = int(input())
    data = [input() for _ in range(m)]
    data = sorted(data, reverse = True)
    flag = True
    for x in data:
        if tree.find(x):
            flag = False
            break
        tree.insert(x)
    print('YES' if flag else 'NO')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-3.png)



### T28046:词梯

bfs, http://cs101.openjudge.cn/practice/28046/

思路：

直接做。寒假做过这题，现在又用类重写了一遍。

代码：

```python
from collections import deque, defaultdict

class Node:
    def __init__(self, key):
        self.key = key
        self.neighbours = []

    def run(self):
        for i in range(4):
            for c in alphabet:
                if c == self.key[i]: continue
                new_node = self.key[:i] + c + self.key[i + 1:]
                if new_node not in G.nodes.keys(): continue
                self.neighbours.append(G.nodes[new_node])

class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, key):
        self.nodes[key] = Node(key)

    def run(self):
        for x in self.nodes.values(): x.run()

    def solve(self, start, end):
        flag = defaultdict(bool)
        q = deque([(self.nodes[start], start + " ")])
        flag[self.nodes[start]] = True
        while q:
            x, path = q.popleft()
            if x == self.nodes[end]:
                print(path)
                return
            for y in x.neighbours:
                if flag[y]: continue
                flag[y] = True
                q.append((y, path + y.key + " "))
        print("NO")

n = int(input())
a = [input() for _ in range(n)]
start, end = input().split()
if 'A' <= a[0][0] <= 'Z': OP = ord('A')
else: OP = ord('a')
alphabet = [chr(OP + i) for i in range(26)]
G = Graph()
for i in range(n): G.add_node(a[i])
G.run()
G.solve(start, end)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-4.png)



### T51.N皇后

backtracking, https://leetcode.cn/problems/n-queens/

思路：

直接做

代码：

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        ans = []
        flag_column = [True] * n
        flag_left_diagonal = [True] * (2 * n + 1) # (i, j) -> i - j + n
        flag_right_diagonal = [True] * (2 * n + 1) # (i, j) -> i + j

        def convert(nums):
            result = [['.' for j in range(n)] for i in range(n)]
            for i in range(n): result[i][nums[i]] = 'Q'
            return [''.join(map(str, row)) for row in result]

        result = []
        def dfs(x):
            if x == n:
                c = ans[:]
                result.append(convert(c))
                return
            for i in range(n):
                if (not flag_column[i]) or (not flag_left_diagonal[x - i + n]) or (not flag_right_diagonal[x + i]): continue
                ans.append(i)
                flag_column[i] = False
                flag_left_diagonal[x - i + n] = False
                flag_right_diagonal[x + i] = False
                dfs(x + 1)
                flag_column[i] = True
                flag_left_diagonal[x - i + n] = True
                flag_right_diagonal[x + i] = True
                ans.pop()
        dfs(0)
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-5.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>


本次作业较易，学习到了一些算法原理（比如Trie树）。但是本周有其他科目的大作业要截止，所以选做题只做了一部分。大致五一期间或者五一后就能赶上选做题。








