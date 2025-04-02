# Assignment #7: 20250402 Mock Exam

Updated 1624 GMT+8 Apr 2, 2025

2025 spring, Complied by <mark>高景行 数学科学学院</mark>



> **说明：**
>
> 1. **⽉考**：AC<mark>6</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
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

### E05344:最后的最后

http://cs101.openjudge.cn/practice/05344/



思路：

忘了可以用队列写了，考场上写了模拟链表的很奇怪的写法

代码：

```python
n, k = map(int, input().split())
nxt = [i + 1 for i in range(n)]; nxt[n - 1] = 0
pre = [i - 1 for i in range(n)]; pre[0] = n - 1
pos, cnt = 0, 1
while cnt <= n - 1:
    for i in range(k - 1): pos = nxt[pos]
    print(pos + 1, end = " ")
    temp = nxt[pos]
    nxt[pre[pos]] = nxt[pos]
    pre[nxt[pos]] = pre[pos]
    pos = temp
    cnt += 1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image.png)



### M02774: 木材加工

binary search, http://cs101.openjudge.cn/practice/02774/



思路：

直接做，注意要除去0的情况


代码：

```python
n, m = map(int, input().split())
a = [int(input()) for i in range(n)]
l, r, ans = 1, 10000, 0

def check(x):
    counting = 0
    for i in range(n): counting += a[i] // mid
    return counting >= m

while l <= r:
    mid = (l + r) >> 1
    if check(mid): ans = mid; l = mid + 1
    else: r = mid - 1
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-1.png)



### M07161:森林的带度数层次序列存储

tree, http://cs101.openjudge.cn/practice/07161/



思路：

层序遍历刚开始没想到怎么方便地处理，最后没办法了写了比较笨重的两个队列的做法。


代码：

```python
from collections import deque

class Tree:
    def __init__(self, name, son_num):
        self.son = []
        self.name = name
        self.num = son_num

def printer(x):
    for t in x.son: printer(t)
    print(x.name, end = " ")

T = int(input())
for ___ in range(T):
    data = list(input().split())
    q1 = deque()
    for i in range(len(data) >> 1):
        name, son_num = data[i << 1], int(data[i << 1 | 1])
        q1.append(Tree(name, son_num))
    rt = q1.popleft()
    q2 = deque([rt])
    while q1:
        x = q2.popleft()
        for i in range(x.num):
            tmp = q1.popleft()
            x.son.append(tmp)
            q2.append(tmp)
    printer(rt)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-2.png)



### M18156:寻找离目标数最近的两数之和

two pointers, http://cs101.openjudge.cn/practice/18156/

思路：

简单双指针，注意l不能与r相等。这题貌似之前留过。


代码：

```python
m = int(input())
a = sorted(list(map(int, input().split()))); n = len(a)
l, r = 0, n - 1
mn, ans = float("inf"), 0
while l < r:
    if a[l] + a[r] == m:
        print(m)
        exit(0)
    if a[l] + a[r] < m:
        if m - a[l] - a[r] <= mn: mn, ans = m - a[l] - a[r], a[l] + a[r]
        l += 1
    else:
        if a[l] + a[r] - m < mn: mn, ans = a[l] + a[r] - m, a[l] + a[r]
        r -= 1
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-3.png)



### M18159:个位为 1 的质数个数

sieve, http://cs101.openjudge.cn/practice/18159/



思路：

很长时间没写欧拉筛导致考场上写的欧拉筛是错的，后面考场上用埃氏筛法才通过本题。场下才发现正确的欧拉筛写法。本题的各种坑点都写在题面（空格、不包括n）。

代码：

```python
NR = 10005
prime = []
ans = []
flag = [True] * (NR + 1); flag[0] = flag[1] = False
for i in range(2, NR + 1):
    if flag[i]:
        prime.append(i)
        if i % 10 == 1: ans.append(i)
    j = 0
    while j < len(prime) and i * prime[j] <= NR:
        flag[i * prime[j]] = False
        if i % prime[j] == 0: break
        j += 1

T = int(input())
for ___ in range(T):
    n = int(input())
    Flag = False
    print(f"Case{___ + 1}:")
    result = []
    for x in ans:
        if x >= n: break
        result.append(x)
        Flag = True
    if not Flag: print("NULL")
    else: print(" ".join(map(str, result)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-4.png)



### M28127:北大夺冠

hash table, http://cs101.openjudge.cn/practice/28127/

思路：

直接做即可。


代码：

```python
from functools import cmp_to_key

class Team:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.ac = 0
        self.already_ac = []

def cmp(x, y):
    if x.ac != y.ac: return y.ac - x.ac
    if x.total != y.total: return x.total - y.total
    if x.name > y.name: return 1
    return -1

teams = []
names = []
T = int(input())
for ___ in range(T):
    s = input().split(",")
    if s[0] not in names:
        node = Team(s[0])
        teams.append(node)
        names.append(s[0])
    else: node = teams[names.index(s[0])]
    node.total += 1
    if s[1] in node.already_ac: continue
    if s[2] == "yes":
        node.ac += 1
        node.already_ac.append(s[1])
teams = sorted(teams, key = cmp_to_key(cmp))
cnt = 0
for i in range(min(12, len(teams))):
    print(f"{i + 1} {teams[i].name} {teams[i].ac} {teams[i].total}")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](image-5.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

本周要准备期中考试，所以没跟上每日选做。

感觉本次月考较基础，考察基本功。事实证明基本功还不太扎实，还是要熟练各种基本算法的写法（如欧拉筛）。









