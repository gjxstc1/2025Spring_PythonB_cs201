# Assignment #D: 图 & 散列表

Updated 2042 GMT+8 May 20, 2025

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

### M17975: 用二次探查法建立散列表

http://cs101.openjudge.cn/practice/17975/

<mark>需要用这样接收数据。因为输入数据可能分行了，不是题面描述的形式。OJ上面有的题目是给C++设计的，细节考虑不周全。</mark>

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
```



思路：

本题学到了hash的方法，但是本题有不少要点题面都没有提到（比如会碰到相同的keys，此时要返回对应的位置）。

注意list的in是O(n)的，而dict的in可以理解成O(1)的！

代码：

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
nums = [int(i) for i in data[index:index+n]]
steps = [(-1)**i * (i // 2 + 1)**2 for i in range(m << 1)]
hash = [-1] * m
result = []
for x in nums:
    idx = init = x % m
    if hash[idx] == -1 or hash[idx] == x:
        hash[idx] = x
        result.append(idx)
        continue
    cur = 0
    while hash[idx] != -1 and hash[idx] != x:
        idx = ((init + steps[cur]) % m + m) % m
        cur += 1
    hash[idx] = x
    result.append(idx)
print(*result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image.png)



### M01258: Agri-Net

MST, http://cs101.openjudge.cn/practice/01258/

思路：

最小生成树模版，但是又在输入上麻烦（比如多组输入，以及题目里说输入数据可能会被拆分到多行中但是实际并没有发生）。

代码：

```python
def find(x):
    if f[x] == x: return x
    f[x] = find(f[x])
    return f[x]

while True:
    try:
        n = int(input())
        a = [list(map(int, input().split())) for _ in range(n)]
        f = [i for i in range(n)]
        edge = []
        for i in range(n):
            for j in range(i + 1,n):
                edge.append((a[i][j], i, j))
        edge = sorted(edge)
        cnt, ans = 0, 0
        for w, u, v in edge:
            if cnt == n - 1: break
            tu, tv = find(u), find(v)
            if tu == tv: continue
            f[tu] = tv
            cnt += 1
            ans += w
        print(ans)
    except EOFError: break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-1.png)



### M3552.网络传送门旅游

bfs, https://leetcode.cn/problems/grid-teleportation-traversal/

思路：

bfs或dijkstra均可，不剪枝下bfs稍好一点。

代码：

```python
class Solution:
    def minMoves(self, a: List[str]) -> int:
        n, m = len(a), len(a[0])
        sx, sy, ex, ey = 0, 0, n - 1, m - 1
        portals = [[] for i in range(26)]
        for i in range(n):
            for j in range(m):
                if 'A' <= a[i][j] <= 'Z': 
                    portals[ord(a[i][j]) - ord('A')].append((i, j))
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        flag = [[False] * m for i in range(n)]
        q = deque([(sx, sy, 0)])
        flag[sx][sy] = True
        if 'A' <= a[sx][sy] <= 'Z':
            for i, j in portals[ord(a[sx][sy]) - ord('A')]:
                if (i, j) != (sx, sy):
                    q.append((i, j, 0))
                    flag[i][j] = True
        while q:
            x, y, d = q.popleft()
            if (x, y) == (ex, ey): return d
            for nx, ny in directions:
                tx, ty = x + nx, y + ny
                if 0 <= tx < n and 0 <= ty < m and not flag[tx][ty] and a[tx][ty] != '#':
                    q.append((tx, ty, d + 1))
                    flag[tx][ty] = True
                    if 'A' <= a[tx][ty] <= 'Z':
                        for i, j in portals[ord(a[tx][ty]) - ord('A')]:
                            if (i, j) != (tx, ty) and not flag[i][j]:
                                q.append((i, j, d + 1))
                                flag[i][j] = True
        return -1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-2.png)



### M787.K站中转内最便宜的航班

Bellman Ford, https://leetcode.cn/problems/cheapest-flights-within-k-stops/

思路：

由于加入了中转次数<=k的限制，所以Bellman Ford非常合适（只松弛k次），$O(nk)$。

也可以用队列优化的Bellman Ford，即SPFA算法。如果没有k次的限制，朴素的Bellman Ford时间复杂度$O(nm)$，而SPFA时间复杂度在随机图上显著好于Bellman Ford，只是最差情况仍是$O(nm)$。这两个算法比Dijkstra（$O(mlogn)$）优势的地方是可以处理负边权。以下给出这两个代码。

代码：

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        edge = [[] for i in range(n)]
        for u, v, w in flights: edge[u].append((v, w))
        q = deque([(src, 1, 0)])
        inqueue, d = [[False] * (k + 3) for i in range(n)], [[float("inf")] * (k + 3) for i in range(n)]
        inqueue[src][1], d[src][1] = True, 0
        while q:
            x, num, dis = q.popleft()
            if num >= k + 2: continue # 已经中转k个点了，不能再往下走任何一步了
            inqueue[x][num] = False
            for y, w in edge[x]:
                if d[y][num + 1] <= d[x][num] + w: continue
                d[y][num + 1] = d[x][num] + w
                if inqueue[y][num + 1]: continue
                inqueue[y][num + 1] = True
                q.append((y, num + 1, d[y][num + 1]))
        ans = min(d[dst])
        return -1 if ans == float("inf") else ans
```

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        d = [float("inf")] * n; d[src] = 0
        for i in range(k + 1):
            tmp = d[:]
            flag = False
            for u, v, w in flights:
                if tmp[v] <= d[u] + w: continue
                tmp[v] = d[u] + w
                flag = True
            if not flag: break
            d = tmp[:]
        return -1 if d[dst] == float("inf") else d[dst]
```


代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-3.png)



### M03424: Candies

Dijkstra, http://cs101.openjudge.cn/practice/03424/

思路：

差分约束，把$X_i<=X_j+c$看成$j$向$i$连边，权重为$c$（类比$d[i]<=d[j]+w$）。如果没有起点和终点的要求，则可以建立超级源点（与每个点的边权为0）再跑最短路。

若图中存在负环，则给定的差分约束系统无解（判断方式是某点最短路经过>=n+1条边），否则$d[i]$就是一组解，并且可以平移。本题数据保证不用判断负环。

代码：

```python
import heapq
n, m = map(int, input().split())
edge = [[] for _ in range(n + 1)]
for _ in range(m):
    x, y, w = map(int, input().split())
    #d[y]<=d[x]+w:
    edge[x].append((y, w))

start, end = 1, n
d = [float('inf')] * (n + 1)
flag = [False] * (n + 1)
q = []; heapq.heappush(q, (0, start))
d[start] = 0
while q:
    dis, x = heapq.heappop(q)
    if x == end: break
    if flag[x]: continue
    flag[x] = True
    for y, w in edge[x]:
        if d[y] <= d[x] + w: continue
        d[y] = d[x] + w
        heapq.heappush(q, (d[y], y))
print(d[end])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-4.png)



### M22508:最小奖金方案

topological order, http://cs101.openjudge.cn/practice/22508/

思路：

拓扑排序即可。

代码：

```python
from collections import deque
n, m = map(int, input().split())
ans = 100 * n
edge = [[] for _ in range(n)]
degree = [0] * n
for _ in range(m):
    x, y = map(int, input().split())
    edge[y].append(x) # y -> x
    degree[x] += 1

q = deque()
for i in range(n):
    if not degree[i]:
        q.append((i, 0))
while q:
    x, money = q.popleft()
    ans += money
    for y in edge[x]:
        if not degree[y]: continue
        degree[y] -= 1
        if not degree[y]: q.append((y, money + 1))
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>


![alt text](image-5.png)


## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

本次作业难度不大，但是有的题目输入较麻烦。

每日选做赶上了现在的进度。









