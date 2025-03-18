# Assignment #5: 链表、栈、队列和归并排序

Updated 1348 GMT+8 Mar 17, 2025

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

### LC21.合并两个有序链表

linked list, https://leetcode.cn/problems/merge-two-sorted-lists/

思路：
没想到题解的巧妙递归。直接做也很简单。


代码：

```python
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        p = list1
        q = list2
        current = ListNode()
        st = current
        while p or q:
            if p and q:
                if p.val < q.val: 
                    current.next = p
                    current = p
                    p = p.next
                else: 
                    current.next = q
                    current = q
                    q = q.next
                continue
            if p: 
                current.next = p
                current = p
                p = p.next
            else: 
                current.next = q
                current = q
                q = q.next
        return st.next
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image.png)



### LC234.回文链表

linked list, https://leetcode.cn/problems/palindrome-linked-list/

<mark>请用快慢指针实现。</mark>

快慢指针是巧妙的做法，可迅速找到一半（向下取整）的位置，然后反转后半部分链表即可

代码：

```python
class Solution(object):
    def isPalindrome(self, head):
        if not head or not head.next: return True
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # fast到倒数第二或最后， slow到一半。反转后面:
        previous = None
        while slow:
            new = slow.next
            slow.next = previous
            previous = slow
            slow = new
        while head and previous: # previous是最后一个元素
            if head.val != previous.val: return False
            head, previous = head.next, previous.next
        return True
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-1.png)



### LC1472.设计浏览器历史记录

doubly-lined list, https://leetcode.cn/problems/design-browser-history/

<mark>请用双链表实现。</mark>

本题学到了类的实例不能让self自身完成赋值更改，所以必须要有实例属性作为指针而不能是实例本身作为指针。


代码：

```python
class DualLink():
    def __init__(self, homepage, nxt = None, pre = None):
        self.val = homepage
        self.next = nxt
        self.pre = pre

class BrowserHistory(object):

    def __init__(self, homepage):
        self.status = DualLink(homepage = homepage)
        """
        :type homepage: str
        """
        

    def visit(self, url):
        new_node = DualLink(url)
        self.status.next = new_node
        new_node.pre = self.status
        self.status = new_node
        """
        :type url: str
        :rtype: None
        """
        

    def back(self, steps):
        count = 0
        node = self.status
        while node.pre and count < steps:
            count += 1
            node = node.pre
        self.status = node
        return node.val
        """
        :type steps: int
        :rtype: str
        """
        

    def forward(self, steps):
        """
        :type steps: int
        :rtype: str
        """
        count = 0
        node = self.status
        while node.next and count < steps:
            count += 1
            node = node.next
        self.status = node
        return node.val
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-2.png)



### 24591: 中序表达式转后序表达式

stack, http://cs101.openjudge.cn/practice/24591/

思路：
学习了Shunting Yard算法。


代码：

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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-3.png)



### 03253: 约瑟夫问题No.2

queue, http://cs101.openjudge.cn/practice/03253/

<mark>请用队列实现。</mark>

学了rotate用法:

##### 正数参数：如果传入正数 n，则表示将队列中的元素向右旋转 n 步。即队列的最后 n 个元素会被移到队列前面

##### 负数参数：如果传入负数 -n，则表示将队列中的元素向左旋转 n 步。即队列的前 n 个元素会被移到队列后面

(也可用popleft，appendleft， pop， append模拟)

代码：

```python
from collections import deque

def josephus(n, p, m):
    q = deque(range(1, n + 1))
    result = []
    q.rotate(- (p - 1))
    while q:
        q.rotate(- (m - 1))
        result.append(q.popleft())
    return result
while True:
    n, p, m = map(int, input().split())
    if n == p == m == 0: break
    print(",".join(map(str, josephus(n, p, m))))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-4.png)



### 20018: 蚂蚁王国的越野跑

merge sort, http://cs101.openjudge.cn/practice/20018/

思路：
求逆序对数模版题。


代码：

```python
n = int(input())
a = list(reversed([int(input()) for _ in range(n)]))
ans = 0

def merge(L, R):
    if L == R: return [a[L]]
    global ans
    mid = (L + R) >> 1
    b, c = merge(L, mid), merge(mid + 1, R)
    l, r = 0, 0
    result = []
    while l < len(b) or r < len(c):
        if l == len(b):
            result.append(c[r])
            r += 1
        elif r == len(c):
            result.append(b[l])
            l += 1
        elif b[l] <= c[r]:
            result.append(b[l])
            l += 1
        else:
            ans += len(b) - l
            result.append(c[r])
            r += 1
    return result

merge(0, n - 1)
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-5.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

本周学习了很多新的想法，很有收获。

紧跟每日选做。

感觉上周的模拟神经网络很有意思。









