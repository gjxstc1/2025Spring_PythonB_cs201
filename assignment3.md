# Assignment #3: 惊蛰 Mock Exam

Updated 1641 GMT+8 Mar 5, 2025

2025 spring, Complied by <mark>高景行 数学科学学院</mark>



> **说明：**
>
> 1. **惊蛰⽉考**：AC<mark>5</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
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

### E04015: 邮箱验证

strings, http://cs101.openjudge.cn/practice/04015



思路：
直接做，可能.在@前面


代码：

```python
while True:
    try:
        s = input().strip().rstrip()
        if s.count("@") != 1:
            print("NO")
            continue
        id = s.index("@")
        if (s[0] in ["@", "."]) or (s[-1] in ["@", "."]):
            print("NO")
            continue
        b = s[id:]
        if (id >= 1 and s[id - 1] == '.') or (s[id + 1] == '.') or ("." not in b):
            print("NO")
            continue
        print("YES")
    except EOFError: break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](image.png)




### M02039: 反反复复

implementation, http://cs101.openjudge.cn/practice/02039/



思路：
直接做


代码：

```python
n = int(input())
s = list(input())
m = len(s) // n
result = []
for i in range(m): # i * n ~ i * n + n - 1
    if i % 2 == 0: result.append(s[i * n: i * n + n])
    else: result.append(s[i * n + n - 1: i * n - 1: -1])
ans = [column for column in zip(*result)]
for col in ans:
    for x in col: print(x, end = "")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](image-1.png)




### M02092: Grandpa is Famous

implementation, http://cs101.openjudge.cn/practice/02092/



思路：
看懂题直接做。


代码：

```python
from functools import cmp_to_key

def cmp(x, y):
    if x[0] != y[0]: return y[0] - x[0]
    return x[1] - y[1]
while True:
    n, m = map(int, input().split())
    if n == m == 0: break
    a = [[0, i] for i in range(10001)]
    for i in range(n):
        tmp = map(int, input().split())
        for x in tmp: a[x][0] += 1
    a = sorted(a, key = cmp_to_key(cmp))
    it = 1
    while it <= 10000 and a[it][0] == a[1][0]:
        print(a[it][1], end = " ")
        it += 1
    print()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-2.png)



### M04133: 垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/



思路：
之前做过的题目


代码：

```python
d = int(input())
n = int(input())
x, y, z = [], [], []
for i in range(n):
    tmpx, tmpy, tmpz = map(int, input().split())
    x.append(tmpx); y.append(tmpy); z.append(tmpz)
ans = 0
cnt = 0
for i in range(1025):
    for j in range(1025):
        this = 0
        for k in range(n):
            if abs(i - x[k]) <= d and abs(j - y[k]) <= d:
                this += z[k]
        if this > ans:
            ans = this; cnt = 1
        elif this == ans: cnt += 1
print(cnt, ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](image-3.png)




### T02488: A Knight's Journey

backtracking, http://cs101.openjudge.cn/practice/02488/



思路：
直接搜索即可，本以为要用Warnsdorff，但发现直接搜索就不超时


代码：

```python
import sys
sys.setrecursionlimit(1 << 30)

def dfs(x, y, z):
    global ans
    global is_exit
    if z == n * m:
        is_exit = True
        return
    for nx, ny in directions:
        tx, ty = x + nx, y + ny
        if 0 <= tx < n and 0 <= ty < m and not flag[tx][ty]:
            flag[tx][ty] = True
            prior_one = ans
            ans += chr(ord('A') + ty) + str(tx + 1)
            dfs(tx, ty, z + 1)
            if is_exit: return
            flag[tx][ty] = False
            ans = prior_one

directions = [(-1, -2), (1, -2), (-2, -1), (2, -1),  (-2, 1), (2, 1), (-1, 2), (1, 2)]
T = int(input())
for case in range(T):
    n, m = map(int, input().split())
    flag = [[False] * m for i in range(n)]
    is_exit = False
    ans = ""
    for j in range(m):
        for i in range(n):
            ans = chr(j + ord('A')) + str(i + 1)
            flag[i][j] = True
            dfs(i, j, 1)
            flag[i][j] = False
            if is_exit: break
        if is_exit: break
    print(f"Scenario #{case + 1}:")
    print(ans if is_exit else "impossible")
    print()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](image-4.png)




### T06648: Sequence

heap, http://cs101.openjudge.cn/practice/06648/



思路：
考场思路是用堆，每拿出队首就枚举 某一位换成下一个数，其余数不变 的情况。关键是每次传入100个数会超内存。并且还有一个问题是可能会重复加入到堆中，而如果要用visited记录则会显著超内存

同时考场上一直在想n很小（$<2^{11}$），说明最多同时10个数列取到第二个数，不过完全没用上。

看了题解学习了方法，每次相当于把两个数列合并成一个：把两个数列对应求和最小的n个数形成新列表（因为总共要找的是最小的n个），这样新列表就和原来普通的数列的作用是一样的，进而转换成m-1个数列。以此类推做下去，有点像归并排序。每次只用向堆里传入2个数表示位置，节省内存。
同时去重可以让堆初始就含有所有的a[i]+b[0]，每次拿出的最小值只用变化b数列即可。

代码：

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



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](image-5.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

本次月考暴露了很多问题：对堆不熟悉，想了很久才想到最后一题用堆的MLE做法。同时对之前学过的算法不太熟悉，缺乏联想能力（比如最后一题做法有点像对两个数列的归并，但考场上没有往这方面想）。
之后要多练习。

数算2025spring每日选做全部AC。
