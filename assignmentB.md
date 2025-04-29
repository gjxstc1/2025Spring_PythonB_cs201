# Assignment #B: 图为主

Updated 2223 GMT+8 Apr 29, 2025

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

### E07218:献给阿尔吉侬的花束

bfs, http://cs101.openjudge.cn/practice/07218/

思路：

著名小说Flowers for Algernon。直接做

代码：

```python
from collections import deque
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
T = int(input())
for ___ in range(T):
    n, m = map(int, input().split())
    a = [list(input()) for _ in range(n)]
    flag = [[False] * m for _ in range(n)]
    for i in range(n):
        if 'S' in a[i]: sx, sy = i, a[i].index('S')
        if 'E' in a[i]: ex, ey = i, a[i].index('E')
    q = deque([(sx, sy, 0)])
    flag[sx][sy] = True
    ans = -1
    while q:
        x, y, z = q.popleft()
        if (x, y) == (ex, ey):
            ans = z
            break
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and a[nx][ny] != '#' and not flag[nx][ny]:
                flag[nx][ny] = True
                q.append((nx, ny, z + 1))
    print(ans if ans != -1 else 'oop!')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image.png)



### M3532.针对图的路径存在性查询I

disjoint set, https://leetcode.cn/problems/path-existence-queries-in-a-graph-i/

思路：

这题如果直接建图反而超时（虽然复杂度我觉得不超时但还是超时了），后来才发现nums单调，然后两个数不在同一类等价于中间出现一次断崖（nums[i] - nums[i - 1] > maxDiff）。

代码：

```python
class Solution:
    def pathExistenceQueries(self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]) -> List[bool]:
        label = [0] * n
        for i in range(1, n): label[i] = label[i - 1] + (nums[i] - nums[i - 1] > maxDiff)
        result = [(label[x] == label[y]) for x, y in queries]
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-1.png)



### M22528:厚道的调分方法

binary search, http://cs101.openjudge.cn/practice/22528/

思路：

直接做

代码：

```python
import math
P = 1000000000

def check(x):
    return x / P * a[idx] + 1.1 ** (x / P * a[idx]) >= 85

a = sorted(list(map(float, input().split())), reverse=True)
n = len(a); idx = math.ceil(n * 0.6) - 1
threshold = a[idx]
l, r, ans = 0, 1000000000, 0
while l <= r:
    mid = (l + r) >> 1
    if check(mid): r, ans = mid - 1, mid
    else: l = mid + 1
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-2.png)



### Msy382: 有向图判环 

dfs, https://sunnywhy.com/sfbj/10/3/382

思路：

本题可以直接做，不用拓扑排序（复杂度应该相同）。

代码：

```python
def dfs(x):
    if visited[x]: return True
    visited[x] = True
    for y in edge[x]:
       if dfs(y): return True
    visited[x] = False
    return False

n, m = map(int, input().split())
edge = [[] for _ in range(n)]
visited = [False for _ in range(n)]
for i in range(m):
    u, v = map(int, input().split())
    edge[u].append(v)

ans = False
for i in range(n):
    if dfs(i):
        print("Yes")
        exit(0)
print("No")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-3.png)



### M05443:兔子与樱花

Dijkstra, http://cs101.openjudge.cn/practice/05443/

思路：

dijkstra模版题目，直接做，输入输出稍有点麻烦

代码：

```python
import heapq
def dij(start, end):
    if start == end:
        print(chk[start])
        return
    q = []
    heapq.heappush(q, (0, start, chk[start] + "->"))
    d = [float("inf")] * n; d[start] = 0
    flag = [False] * n
    while q:
        dis, u, ans = heapq.heappop(q)
        if u == end:
            print(ans)
            return
        if flag[u]: continue # 重要
        flag[u] = True # 重要
        for v, w in edge[u]:
            if dis + w >= d[v]: continue
            d[v] = dis + w
            heapq.heappush(q, (d[v], v, ans + "(" + str(w) + ")" + "->" + chk[v] + ("->" if v != end else "")))

n = int(input()); mp = {}; chk = []
for i in range(n):
    c = input()
    mp[c] = i; chk.append(c)
m = int(input())
edge = [[] for _ in range(n)]
for __ in range(m):
    u, v, w = input().split()
    edge[mp[u]].append((mp[v], int(w)))
    edge[mp[v]].append((mp[u], int(w)))
T = int(input())
for __ in range(T):
    st, ed = input().split()
    st, ed = mp[st], mp[ed]
    dij(st, ed)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-4.png)



### T28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/

思路：

本题寒假做过，学到了一个很“黑盒”的优化方式：因为只用知道是否存在一条路径，故经过的每个点处对所有可能到达的邻居，按照这些邻居下一步可以到达的次邻居数量从小到大排序，先走次邻居少的（避免大的计算量的偷懒？），不太清楚什么时候适用。

代码：

```python
import sys
sys.setrecursionlimit(1 << 20)

def dfs(x, y, d):
    if d == n * n:
        print("success")
        exit(0)
    candidate = []
    for nx, ny in directions:
        tx, ty = x + nx, y + ny
        if 0 <= tx < n and 0 <= ty < n and not flag[tx][ty]:
            tmp = 0
            for dx, dy in directions:
                mx, my = tx + dx, ty + dy
                if 0 <= mx < n and 0 <= my < n and not flag[mx][my]: tmp += 1
            candidate.append((tmp, tx, ty))
    for tmp, wx, wy in sorted(candidate):
        flag[wx][wy] = True
        dfs(wx, wy, d + 1)
        flag[wx][wy] = False
n = int(input())
sx, sy = map(int, input().split())
if n % 2 and (sx + sy) % 2:
    print("fail")
    exit(0)
directions = [[2, -1], [2, 1], [-2, 1], [-2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2]]
flag = [[False] * n for _ in range(n)]; flag[sx][sy] = True
dfs(sx, sy, 1)
print("fail")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-5.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>


这周是最忙的一周，有一个其他课程的大作业要赶，所以还没补做本周每日选做








