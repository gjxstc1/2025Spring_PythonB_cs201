# Cheatsheet

## 0 修改递归深度

```python
import sys
sys.setrecursionlimit(1 << 30)
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

## 2 树状数组

## 3 线段树

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

## 10 单源最短路

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
        flag[x][y] = True #重要！！
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
        flag[u] = True
        for v, w in edge[u]:
            if dis + w >= d[v]: continue
            d[v] = dis + w
            heapq.heappush(q, (d[v], v, ans + "(" + str(w) + ")" + "->" + chk[v] + ("->" if v != end else "")))
```

## 11 KMP

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

## 12 康托展开

```python

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
rt = (parent - children).pop() # 集合A\B
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

## 17 


