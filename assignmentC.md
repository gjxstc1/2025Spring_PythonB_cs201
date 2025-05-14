# Assignment #C: 202505114 Mock Exam

Updated 1518 GMT+8 May 14, 2025

2025 spring, Complied by <mark>高景行 数学科学学院</mark>



> **说明：**
>
> 1. **⽉考**：AC<mark>5</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
> 2. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 3. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E06364: 牛的选举

http://cs101.openjudge.cn/practice/06364/

思路：

直接做

代码：

```python
from functools import cmp_to_key

class cow:
    def __init__(self, id, a, b):
        self.id = id
        self.a = a
        self.b = b

def cmp1(x, y):
    return y.a - x.a

def cmp2(x, y):
    return y.b - x.b

n, K = map(int, input().split())
q = []
for i in range(n):
    x, y = map(int, input().split())
    q.append(cow(i + 1, x, y))
q = sorted(q, key = cmp_to_key(cmp1))[:K]
q = sorted(q, key = cmp_to_key(cmp2))
print(q[0].id)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image.png)



### M04077: 出栈序列统计

http://cs101.openjudge.cn/practice/04077/

思路：

本题有结论（Catalan数），可以直接做

代码：

```python
n = int(input())
c = [0] * 16; c[0] = c[1] = 1
for x in range(2, n + 1):
    for i in range(x): c[x] += c[i] * c[x - 1 - i]
print(c[n])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>


![alt text](image-1.png)


### M05343:用队列对扑克牌排序

http://cs101.openjudge.cn/practice/05343/

思路：

直接做

代码：

```python
from collections import deque
n = int(input())
a = list(input().split())
q = [deque() for i in range(10)]
for i in range(1, 10):
    for x in a:
        if int(x[1]) == i: q[i].append(x)
b = []
for i in range(1, 10):
    print(f"Queue{i}:", end = "")
    while q[i]:
        x = q[i].popleft()
        print(x, end = " ")
        b.append(x)
    print()
for i in range(1, 5):
    for x in b:
        if x[0] == chr(ord('A') + i - 1): q[i].append(x)
c = []
for i in range(1, 5):
    print(f"Queue{chr(ord('A') + i - 1)}:", end = "")
    while q[i]:
        x = q[i].popleft()
        print(x, end = " ")
        c.append(x)
    print()
print(*c)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-2.png)



### M04084: 拓扑排序

http://cs101.openjudge.cn/practice/04084/

思路：

要求排序有点奇怪，我的做法是用heapq。

代码：

```python
import heapq
n, m = map(int, input().split())
edge = [[] for i in range(n + 1)]
degree = [0] * (n + 1)
for i in range(m):
    x, y = map(int, input().split())
    edge[x].append(y)
    degree[y] += 1
q = []
for i in range(1, n + 1):
    edge[i] = sorted(edge[i])
    if not degree[i]: heapq.heappush(q, i)

while q:
    x = heapq.heappop(q)
    print(f"v{x} ", end = "")
    for y in edge[x]:
        if not degree[y]: continue
        degree[y] -= 1
        if not degree[y]: heapq.heappush(q, y)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-3.png)



### M07735:道路

Dijkstra, http://cs101.openjudge.cn/practice/07735/

思路：

本题在考场上陷入了错误的思路，一个多小时没做出来。错误的思路是：以为是沿用上题思想，先做拓扑排序（这样得到的序列满足无后效性，后面的状态不会改变前面的状态），然后再用一个dp（因为多了一层通行费的约束，我以为用dp处理：dp[x][j]表示从1到达节点x，通行费花了j的最短路长度）。但是关键问题是原图可能有一堆圈，这样无法拓扑排序。考场下才知道这样有其他约束的题目也可以用Dijkstra（复杂度O(mlogn)，和费用K无关）等最短路算法

代码：

```python
import heapq
K = int(input())
n = int(input())
m = int(input())
edge = [[] for i in range(n + 1)]
for i in range(m):
    x, y, length, toll = map(int, input().split())
    if y == x: continue
    edge[x].append((y, length, toll))
dp = [[float("inf")] * (K + 1) for i in range(n + 1)]
q = []
flag = [[False] * (K + 1) for i in range(n + 1)]
heapq.heappush(q, (0, 0, 1))
dp[1][0] = 0
while q:
    total, cost, x = heapq.heappop(q)
    if x == n:
        print(total)
        exit(0)
    if flag[x][cost]: continue
    flag[x][cost] = True
    for y, length, toll in edge[x]:
        if cost + toll > K: continue
        if total + length < dp[y][cost + toll]:
            dp[y][cost + toll] = total + length
            heapq.heappush(q, (dp[y][cost + toll], cost + toll, y))
print(-1)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-4.png)



### T24637:宝藏二叉树

dp, http://cs101.openjudge.cn/practice/24637/

思路：

注意读题，本题较易。本题直接用dp即可，dp[x][0/1]表示从x往下走，取x/不取x的数值，得到的最大分数。

代码：

```python
from collections import deque
n = int(input())
a = [0] + list(map(int, input().split()))
if n == 1:
    print(a[1])
    exit(0)
m = (n + 1) >> 1
f = [[0, 0] for i in range(2 * n + 2)]
q = deque()
for i in range(m, n + 1):
    f[i][1] = a[i]
    q.append(i)

while q:
    x = q.popleft()
    father = x >> 1
    if not father: continue
    f[father][1] = a[father] + f[father << 1][0] + f[father << 1 | 1][0]
    f[father][0] = max(f[father << 1][1], f[father << 1][0]) + max(f[father << 1 | 1][1], f[father << 1 | 1][0])
    q.append(father)
print(max(f[1][0], f[1][1]))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-5.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>


本次月考发挥很差，主要原因是之前几周比较忙，练习少。对一些常见模型也不够敏感（比如最短路模型可以多一个通用费的约束条件，仍然可以用Dijkstra）。本周也在尽量补之前的每日选做，应该下周能基本补上。








