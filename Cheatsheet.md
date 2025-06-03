# Cheatsheet

| 方法名                                                     | 用途说明                        |
| ---------------------------------------------------------- | ------------------------------- |
| `__eq__(self, other)` | `==` | 判断相等 |
| `__ne__(self, other)` | `!=` | 判断不相等 |
| `__lt__(self, other)` | `<` | 判断是否小于 |
| `__le__(self, other)` | `<=` | 判断是否小于等于 |
| `__gt__(self, other)` | `>` | 判断是否大于 |
| `__ge__(self, other)` | `>=` | 判断是否大于等于 |

**二叉搜索树：若它的左子树不空，则左子树上所有结点的值均小于它的根结点的值； 若它的右子树不空，则右子树上所有结点的值均大于它的根结点的值。**

**堆的删除用数组记录，懒删除。**

## 0 修改递归深度/常见代码

```python
import sys
sys.setrecursionlimit(1 << 30)
#---

from collections import defaultdict
from collections import deque
#---

from functools import lru_cache
@lru_cache(maxsize = None)
def dfs(x, y): pass
#---

bisect.bisect_left(a, target) # = lower_bound，返回第一个数组a中>=target的位置,没有就返回len(a)；
bisect.bisect_right(a, target) # = upper_bound，返回第一个数组a中>target的位置,没有就返回len(a)
#---

```

## 1 并查集（+set使用方法）

```python
def find(x):
    if f[x] == x: return x
    f[x] = find(f[x])
    return f[x]
cnt = 0
while True:
    n, m = map(int, input().split())
    if n == m == 0: break
    cnt += 1
    f = [i for i in range(n + 1)]
    for i in range(m):
        x, y = map(int, input().split())
        f[find(x)] = find(y)
    s = set()
    for i in range(1, n + 1): s.add(find(i))
    print(f"Case {cnt}: {len(s)}")
```

## 2 Trie树（是否有前缀）

```python
class Node:
    def __init__(self):
        self.children = {}

class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, nums):
        cur = self.root
        for x in nums:
            if x not in cur.children: cur.children[x] = Node()
            cur = cur.children[x]

    def search(self, nums):
        cur = self.root
        for x in nums:
            if x not in cur.children: return False
            cur = cur.children[x]
        return True
n = int(input())
tree = Trie()
flag = True
a = sorted([str(input()) for _ in range(n)], reverse = True)
for c in a:
    if not flag: continue
    if tree.search(c): flag = False
    else: tree.insert(c)
print("YES" if flag else "NO")
```

## 3 类重定义"<"(方便用heapq)

```python
import heapq
import sys
output = sys.stdout.write
class Node:
    def __init__(self, x, id):
        self.x = x
        self.id = id
    def __lt__(self, other): # 定义小于号！
        return self.x > other.x
ans = ""
q = [] # 和列表相同，只是语法上不同; q[0]是最小值！小根堆！
n = int(input())
flag = [False] * n
a = list(map(int, input().split()))
b = list(map(int, input().split()))
now = 0; cnt = 0
for i in range(n):
    now += a[i]
    if now >= b[i]:
        now -= b[i]
        heapq.heappush(q, Node(b[i], i))
        flag[i] = True
        cnt += 1
    elif q and b[i] < q[0].x:
        tmp = heapq.heappop(q) # 先返回值后pop
        flag[tmp.id] = False
        flag[i] = True
        now += tmp.x - b[i]
        heapq.heappush(q, Node(b[i], i))
ans += str(cnt) + "\n"
ans += " ".join(map(str, [i + 1 for i in range(n) if flag[i]]))
output(ans)
```

## 4 st表

```python
def calculate(l, r):
    tmp = lg[r - l + 1]
    if st[l][tmp] <= st[r - (1 << tmp) + 1][tmp]: return pos[l][tmp]
    return pos[r - (1 << tmp) + 1][tmp]
T = int(input())
lg = [0] * 20
for i in range(1, 16): lg[i] = lg[i - 1] + ((1 << lg[i - 1]) == i)
for i in range(1, 16): lg[i] -= 1
st = [[0] * 6 for _ in range(16)]
pos = [[0] * 6 for _ in range(16)]
for ___ in range(T):
    x, K = input().split()
    n = len(x); K = n - int(K)
    ans = ""
    for i in range(n): st[i][0] = int(x[i]); pos[i][0] = i
    for k in range(1, 6):
        for j in range(n - (1 << k) + 2):
            st[j][k] = min(st[j][k - 1], st[j + (1 << (k - 1))][k - 1])
            if st[j][k - 1] <= st[j + (1 << (k - 1))][k - 1]: pos[j][k] = pos[j][k - 1]
            else: pos[j][k] = pos[j + (1 << (k - 1))][k - 1]
    cur = 0
    for i in range(K):
        cur = calculate(cur, n - K + i)
        ans += x[cur]
        cur += 1
    print(ans)
```

## 5 约瑟夫问题

```python
while True:
    n, m = map(int, input().split())
    if not n and not m: break
    a = [0] * (n + 1)
    a[1] = 0
    for i in range(2, n + 1): a[i] = (a[i - 1] + m) % i
    a[n] += 1
    print(a[n])
```

## 6 堆结构实现

### 小根堆（顶为最小值）

```python
class Min_Heap:
    def __init__(self):
        self.heap = []
    def father(self, index): return (index - 1) >> 1
    def left_son(self, index): return index << 1 | 1
    def right_son(self, index): return (index + 1) << 1
    def insert(self, key):
        self.heap.append(key)
        current_index = len(self.heap) - 1
        while current_index and self.heap[self.father(current_index)] > self.heap[current_index]:
            self.heap[self.father(current_index)], self.heap[current_index] = self.heap[current_index], self.heap[self.father(current_index)]
            current_index = self.father(current_index)
    def query_min(self): # 输出并删除
        if not len(self.heap): return None
        if len(self.heap) == 1: return self.heap.pop()
        rt = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.min_heapify(0)
        return rt
    def min_heapify(self, index):
        smallest = index
        l = self.left_son(index)
        r = self.right_son(index)
        if l < len(self.heap) and self.heap[l] < self.heap[smallest]: smallest = l
        if r < len(self.heap) and self.heap[r] < self.heap[smallest]: smallest = r
        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self.min_heapify(smallest)
m = int(input())
q = Min_Heap()
for ___ in range(m):
    INPUT = list(map(int, input().split()))
    if INPUT[0] == 1: q.insert(INPUT[1])
    else: print(q.query_min())
```

## 7 逆序对数

```python
n = int(input())
ans = 0
a = list(map(int, input().split()))

def g(L, R):
    global ans
    if L == R: return [a[L]]
    mid = (L + R) >> 1
    b = g(L, mid) + [float("inf")]; c = g(mid + 1, R) + [float("inf")]
    l, r = 0, 0
    d = []
    while l <= mid - L or r <= R - mid - 1:
        if b[l] < c[r]: d.append(b[l]); l += 1
        else: d.append(c[r]); r += 1; ans += mid - L - l + 1
    return d
g(0, n - 1)
print(ans)
```

## 8 树的存储与前中后序遍历

```python
def preorder_traversal(rt):
    print(rt.val, end = "")
    if rt.left:
        preorder_traversal(rt.left)
    if rt.right:
        preorder_traversal(rt.right)

def inorder_traversal(rt):
    if rt.left:
        inorder_traversal(rt.left)
    print(rt.val, end = "")
    if rt.right:
        inorder_traversal(rt.right)

def postorder_traversal(rt):
    if rt.left:
        postorder_traversal(rt.left)
    if rt.right:
        postorder_traversal(rt.right)
    print(rt.val, end = "")

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def parse(index):
    if index >= n: return None, index
    if s[index] == '*': return None, index + 1
    newnode = TreeNode(s[index])
    index += 1
    if index < n and s[index] == '(':
        newnode.left, index = parse(index + 1)
        if s[index] == ',': newnode.right, index = parse(index + 1)
    if index < n and s[index] == ')': index += 1
    return newnode, index

T = int(input())
for ___ in range(T):
    s = input(); n = len(s)
    RT, __ = parse(0)
    preorder_traversal(RT)
    print()
    inorder_traversal(RT)
    print()
```

## 9 最小生成树

```python
def find(x):
    if f[x] == x: return x
    f[x] = find(f[x])
    return f[x]

n, m = map(int, input().split())
a = []
for _ in range(m): a.append(list(map(int, input().split())))
a = sorted(a, key = lambda t: t[2])
f = [i for i in range(n + 1)]
edge, ans = 0, 0
for [u, v, w] in a:
    tu, tv = find(u), find(v)
    if tu == tv: continue
    f[find(tu)] = find(tv)
    edge += 1; ans = w
    if edge == n - 1: break
print(edge, ans)
```

## 10 单源最短路（都是Dijkstra），复杂度O(mlogn)

```python
import heapq
def solve(sx, sy, ex, ey):
    q = []
    if a[sx][sy] == '#' or a[ex][ey] == '#':
        print("NO")
        return
    flag = [[False] * m for _ in range(n)] # 是否 in 过 q
    d = [[float('inf')] * m for _ in range(n)]
    heapq.heappush(q, (0, sx, sy))
    d[sx][sy] = 0
    while q:
        dist, x, y = heapq.heappop(q)
        if x == ex and y == ey: break
        if flag[x][y]: continue
        flag[x][y] = True #重要， flag表示曾经到过且离开了，不会再到了
        for nx, ny in directions:
            tx, ty = x + nx, y + ny
            if 0 <= tx < n and 0 <= ty < m and a[tx][ty] != "#":
                if abs(int(a[x][y]) - int(a[tx][ty])) + dist >= d[tx][ty]: continue
                d[tx][ty] = abs(int(a[x][y]) - int(a[tx][ty])) + dist
                heapq.heappush(q, (d[tx][ty], tx, ty))
    print("NO" if d[ex][ey] == float('inf') else d[ex][ey])
```

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
        if flag[u]: continue
        flag[u] = True # 在循环前用flag
        for v, w in edge[u]:
            if dis + w >= d[v]: continue
            d[v] = dis + w
            heapq.heappush(q, (d[v], v, ans + "(" + str(w) + ")" + "->" + chk[v] + ("->" if v != end else "")))
```

#### 补：差分约束：

```python
for _ in range(m):
    x, y, w = map(int, input().split())
    #d[y]<=d[x]+w:
    edge[x].append((y, w))
```

```python
q = []
flag = [[False] * (K + 1) for i in range(n + 1)]
heapq.heappush(q, (0, 0, 1)) # dis, cost, x
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

## 12 SPFA（不能提前终止，要最后一块算）

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

## 13 set

```python
n = int(input())
a = []
for ___ in range(n):
    data = input().split()
    a.append(set(map(int, data[1:])))
m = int(input())
for ___ in range(m):
    b = list(map(int, input().split()))
    ans = set(); first = True
    for i in range(n):
        if b[i] == 1:
            if first: ans = a[i].copy(); first = False
            else: ans &= a[i] # 注意要拷贝，如果两个list/set为a=b那么是同一引用，修改一个会改另一个
    for i in range(n):
        if b[i] == -1: ans -= a[i]
    if not ans: print("NOT FOUND"); continue
    print(" ".join(map(str, sorted(ans))))
```

```python
def dfs(x):
    ans = 0
    if l[x] != -1: ans = max(ans, dfs(l[x]))
    if r[x] != -1: ans = max(ans, dfs(r[x]))
    return ans + 1

n = int(input())
l, r = [-1] * (n + 1), [-1] * (n + 1)
rt = -1
children = set()
for _ in range(n):
    i = _ + 1
    l[i], r[i] = map(int, input().split())
    if l[i] != -1: children.add(l[i])
    if r[i] != -1: children.add(r[i])
rt = (set(range(1, n + 1)) - children).pop()
print(dfs(rt))
```

## 14 最大子矩形

```python
import sys
input = sys.stdin.read
data = list(map(int, input().split()))
n = data[0]
a = [[0] * (n + 1)] + [[0] + data[i * n + 1: (i + 1) * n + 1] for i in range(n)]
s = [[0] * (n + 1) for i in range(n + 1)]
for i in range(1, n + 1):
    for j in range(1, n + 1): s[i][j] = s[i][j - 1] + a[i][j]
ans = float("-inf")
for i in range(1, n + 1):
    for j in range(i, n + 1):
        max_global = max_current = s[1][j] - s[1][i - 1]
        for k in range(2, n + 1):
            max_current = max(max_current, 0) + s[k][j] - s[k][i - 1]
            max_global = max(max_global, max_current)
        ans = max(ans, max_global)
print(ans)
```

## 15 中缀转后缀表达式

```python
T = int(input())
while T:
    T -= 1
    s = input(); n = len(s)
    q, result = [], [] # q: 运算符
    precedence = {"+": 0, "-": 0, "*": 1, "/":1}
    pos = 0
    while pos < n:
        if s[pos].isdigit():
            num = ""
            while pos < n and (s[pos].isdigit() or s[pos] == '.'):
                num += s[pos]
                pos += 1
            result.append(num)
            continue
        if s[pos] == "(": q.append(s[pos])
        elif s[pos] in precedence:
            while q and q[-1] in precedence and precedence[q[-1]] >= precedence[s[pos]]: result.append(q.pop())
            q.append(s[pos])
        elif s[pos] == ")":
            while q and q[-1] != "(": result.append(q.pop())
            if q and q[-1] == "(": q.pop()
        pos += 1
    while q: result.append(q.pop())
    print(" ".join(map(str, result)))
```

## 16 欧拉筛法

```cpp
# include <bits/stdc++.h>
# define f(i,a,b) for (int i = a; i <= b; i++)
# define _f(i,a,b) for (int i = a; i >= b; i--)
using namespace std;
const int NR = 1e6;
int n;
long long a[NR + 1];
bool isprime[NR + 1];
int prime[NR + 1], cnt = 0;
int phi[NR + 1], miu[NR + 1];

int main() {
    scanf("%d", &n);
    phi[1] = 1; miu[1] = 1;
    f(i,1,NR) isprime[i] = true;
    isprime[0] = isprime[1] = false;
    f(j,2,NR) {
        if (isprime[j]) {
            prime[++cnt] = j;
            miu[j] = -1;
            phi[j] = j - 1;
        }
        for (int i = 1; i <= cnt && j * prime[i] <= NR; i++) {
            isprime[j * prime[i]] = false;
            if (j % prime[i] == 0) {
                miu[j * prime[i]] = 0;
                phi[j * prime[i]] = phi[j] * prime[i];
                break;
            }
            phi[j * prime[i]] = phi[j] * phi[prime[i]];
            miu[j * prime[i]] = -miu[j];
        }
    }
    return 0;
}
```

## 17 两个堆维护中位数

```python
from collections import defaultdict, deque
import heapq
# 删掉第一个数是不排序下的第一个数

class DualHeap:
    def __init__(self):
        self.q_small, self.q_large = [], []  # small: 从大到小，存较小的一半, 取负号； large: 从小到大，存较大的一半
        self.delay = defaultdict(int)
        self.q_small_size, self.q_large_size = 0, 0 # large size <= small size <= large size + 1,一定是这样的。
                                                    # 另一种方式的反例：add 1, add 2, query
    def update(self, heap):
        if heap == self.q_small:
            while heap and self.delay[-heap[0]]:
                self.delay[-heap[0]] -= 1
                heapq.heappop(heap)
        else:
            while heap and self.delay[heap[0]]:
                self.delay[heap[0]] -= 1
                heapq.heappop(heap)

    def balance(self):
        if self.q_small_size > self.q_large_size + 1:
            self.update(self.q_small)
            top = -heapq.heappop(self.q_small)
            self.q_small_size -= 1; self.q_large_size += 1
            heapq.heappush(self.q_large, top)
        elif self.q_small_size < self.q_large_size:
            self.update(self.q_large)
            top = heapq.heappop(self.q_large)
            self.q_small_size += 1; self.q_large_size -= 1
            heapq.heappush(self.q_small, -top)

    def push(self, x):
        if not self.q_small_size or -self.q_small[0] >= x: heapq.heappush(self.q_small, -x); self.q_small_size += 1
        else: heapq.heappush(self.q_large, x); self.q_large_size += 1
        self.balance()

    def remove(self, x):
        self.delay[x] += 1
        if self.q_small and x <= -self.q_small[0]:
            self.q_small_size -= 1
            if self.q_small and x == -self.q_small[0]: self.update(self.q_small) # 重要的一行，没有竟然过不了。保证中间的两个数是正确的，这样后面在push的时候就是和正确的中间值比较。
        else:
            self.q_large_size -= 1
            if self.q_large and x == self.q_large[0]: self.update(self.q_large) # 重要的一行，没有竟然过不了。保证中间的两个数是正确的，这样后面在push的时候就是和正确的中间值比较。
        self.balance()

    def query(self):
        self.update(self.q_small); self.update(self.q_large)
        if (self.q_large_size + self.q_small_size) & 1: return -self.q_small[0]
        return (-self.q_small[0] + self.q_large[0]) / 2

def main():
    n = int(input())
    q = DualHeap()
    in_queue = deque()
    for i in range(n):
        c = input().split()
        if c[0] == "query":
            x = q.query()
            if x == int(x): print(int(x))
            else: print(x)
        elif c[0] == "del":
            x = in_queue.popleft()
            q.remove(x)
        else:
            x = int(c[1])
            in_queue.append(x)
            q.push(x)
```

```cpp
# include <bits/stdc++.h>
# define f(i,a,b) for (int i = a; i <= b; i++)
# define _f(i,a,b) for (int i = a; i >= b; i--)
using namespace std;
int n;
priority_queue <int> q1;
priority_queue <int, vector<int>, greater<int> > q2;

int main() {
	scanf("%d", &n);
	int x;
	f(i,1,n) {
		cin >> x;
		q1.push(x);
		q2.push(q1.top());
		q1.pop();
		if (i % 2 == 0) {
			q1.push(q2.top());
			q2.pop();
		}
        if (i % 2 == 1) printf("%d\n", q2.top());
    }
    return 0;
}
```

## 18 Bellman Ford（适合处理中转次数<=k）

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

## 19 Tarjan（不考，gc表示对应的大点类别）

```cpp
# include <bits/stdc++.h>
# define f(i,a,b) for (int i = a; i <= b; i++)
# define _f(i,a,b) for (int i = a; i >= b; i--)
# define ll long long
using namespace std;
const int NR = 1e5;
int n, m;

struct Node {
    int to, nxt, w;
} g[2 * NR + 1];

int fe[NR + 1], tmp = 0;
void addedge(int x, int y, int z) {
    g[++tmp] = (Node) {y, fe[x], z};
    fe[x] = tmp;
}

int a[NR + 1];
stack <int> q;
int low[NR + 1], dfn[NR + 1], now = 0;
int gc[NR + 1], num[NR + 1], col = 0;

void tarjan(int x) {
    low[x] = dfn[x] = ++now;
    q.push(x);
    for (int i = fe[x]; i; i = g[i].nxt) {
        int y = g[i].to;
        if (!dfn[y]) {
            tarjan(y);
            low[x] = min(low[x], low[y]);
        }
        else if (!gc[y]) low[x] = min(low[x], dfn[y]);
    }
    if (low[x] == dfn[x]) {
        col++;
        while (q.top() != x) {
            int tt = q.top(); q.pop();
            gc[tt] = col;
            num[col] += a[tt];
        }
        int tt = q.top(); q.pop();
        gc[tt] = col;
        num[col] += a[tt];
    }
}

int u[NR + 1], v[NR + 1];
int in[NR + 1];
int dp[NR + 1];

void tp() {
    queue <int> qq;
    f(i,1,col) {
        if(!in[i]) {
            qq.push(i);
            dp[i] = num[i];
        }
    }
    while (!qq.empty()) {
        int x = qq.front(); qq.pop();
        for (int i = fe[x]; i; i = g[i].nxt) {
            int y = g[i].to;
            dp[y] = max(dp[y], dp[x] + num[y]);
            in[y]--;
            if (!in[y]) qq.push(y);
        }
    }
}

int main() {
    scanf("%d%d", &n, &m);
    f(i,1,n) scanf("%d", a + i);
    f(i,1,m) {
        scanf("%d%d", u + i, v + i);
        addedge(u[i], v[i], 0);
    }
    f(i,1,n) {
        if (!dfn[i]) tarjan(i);
    }
    memset(fe, 0, sizeof(fe));
    tmp = 0;
    f(i,1,m) {
        if (gc[u[i]] != gc[v[i]]) {
            in[gc[v[i]]]++;
            addedge(gc[u[i]], gc[v[i]], 0);
        }
    }
    tp();
    int mx = 0;
    f(i,1,col) mx = max(mx, dp[i]);
    printf("%d\n", mx);
    return 0;
}
```

## 20 拓扑排序（同条件字典序输出）

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

## 21 滑动窗口

```cpp
# include <bits/stdc++.h>
# define f(i,a,b) for (int i = a; i <= b; i++)
# define _f(i,a,b) for (int i = a; i >= b; i--)
using namespace std;
const int NR = 1e6;
int n, k, x;

struct Node {
	int x, id;
} a[NR + 1];
deque <Node> q;

int main() {
	scanf("%d%d", &n, &k);
	f(i,1,n) {
		scanf("%d", &a[i].x);
		a[i].id = i;
	}
	// 求最小值， dq中储存的数单调上升，每存一个数要求比前一个数大，否则前面的数会被单调队列掉
	f(i,1,n) {
		while (!q.empty() && q.front().id <= i - k) q.pop_front();// 窗口长度
		while (!q.empty() && a[i].x <= q.back().x) q.pop_back();
		q.push_back(a[i]);
		if (i < k) continue;
		printf("%d ", q.front().x);
	}
	q.clear();
	puts("");
	// 最大值
	f(i,1,n) {
		while (!q.empty() && q.front().id <= i - k) q.pop_front();
		while (!q.empty() && a[i].x >= q.back().x) q.pop_back();
		q.push_back(a[i]);
		if (i < k) continue;
		printf("%d ", q.front().x);
    }
	return 0;
}
```

```python
from collections import deque
import sys
output = sys.stdout.write
ans = ""
q = deque()
n, k = map(int, input().split())
a = list(map(int, input().split()))
for i in range(n):
    while q and q[0] <= i - k: q.popleft()
    while q and a[i] <= a[q[-1]]: q.pop()
    q.append(i)
    if i < k - 1: continue
    ans += str(a[q[0]]) + " "
ans += "\n"
q.clear()
for i in range(n):
    while q and q[0] <= i - k: q.popleft()
    while q and a[i] >= a[q[-1]]: q.pop()
    q.append(i)
    if i < k - 1: continue
    ans += str(a[q[0]]) + " "
ans += "\n"
output(ans)
```

## 以下不考/具体问题

## 11 KMP（不考）

```python
x = input().strip(); s = input().strip() # s:模式串（匹配串），x：文本串
# 注意如果连着行读入，最后要有.strip()!!!
na, nb = len(x), len(s)
nxt = [0 for __ in range(nb)]
j = 0
for i in range(1, nb):
    while j and s[i] != s[j]: j = nxt[j - 1]
    if s[i] == s[j]: j += 1
    nxt[i] = j
# 上面初始化nxt数组， nxt[i]=j表示最大的j<=i,使模式串s[0~j-1] == s[i-j+1~i]。 代码中所有含nxt的都是nxt[j-1]
j = 0
for i in range(na):
    while j and x[i] != s[j]: j = nxt[j - 1]
    if x[i] == s[j]: j += 1
    if j == nb:
        j = nxt[j - 1]
        print(i - nb + 1 + 1)
print(*nxt)
```

## 22 多序列"排队"（两两合并，只用考虑两个序列排队）

```python
import heapq
T = int(input())
for ___ in range(T):
    m, n = map(int, input().split())
    a = [sorted(list(map(int, input().split()))) for i in range(m)]
    last = a[0]
    for j in range(1, m): # merge a[j], last, 注意要不能重复加入
        q = [(last[0] + a[j][i], [0, i]) for i in range(n)]; heapq.heapify(q)
        new = []
        for case in range(n):
            sm, lst = heapq.heappop(q)
            new.append(sm)
            if lst[0] < n - 1:
                s = sm + last[lst[0] + 1] - last[lst[0]]
                heapq.heappush(q, (s, [lst[0] + 1, lst[1]]))
        last = new
    print(*last)
```

## 23 辅助栈

```python
def main():
    q = [] # 所有猪，stack
    mn = [] # 每次push后的min猪，和q对应 stack
    while True:
        try:
            s = input().split() # 这时s是list
            if s[0] == "pop":
                if q and mn: q.pop(); mn.pop()
            if s[0] == "push":
                x = int(s[1])
                q.append(x)
                if not mn: mn.append(x)
                else: mn.append(min(mn[-1], x))
            if s[0] == "min":
                if q and mn: print(mn[-1])
        except EOFError: break
```
