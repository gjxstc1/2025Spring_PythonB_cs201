# Assignment #4: 位操作、栈、链表、堆和NN

Updated 1203 GMT+8 Mar 10, 2025

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

### 136.只出现一次的数字

bit manipulation, https://leetcode.cn/problems/single-number/



<mark>请用位操作来实现，并且只使用常量额外空间。</mark>

直接异或即可。可以用reduce。

代码：

```python
class Solution(object):
    def singleNumber(self, nums):
        return reduce(lambda x, y: x ^ y, nums)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![alt text](image-1.png)



### 20140:今日化学论文

stack, http://cs101.openjudge.cn/practice/20140/



思路：
这个栈需要想的清楚一些。可以在加入左括号时把这个左括号前面的字符串和这一整体的重复次数放入栈中。


代码：

```python
q = []; s = input()
current_num = 1
current_string = ""
index = 0
while index < len(s):
    if s[index].isalpha():
        current_string = current_string + s[index]
        index += 1
    elif s[index].isdigit():
        this_num = ""
        while s[index].isdigit():
            this_num += s[index]
            index += 1
        current_num = int(this_num)
    elif s[index] =='[':
        q.append((current_string, current_num))
        current_string = ""; current_num = 0
        index += 1
    elif s[index] == ']':
        x, y = q.pop()
        x += current_num * current_string
        current_string = x; current_num = y
        index += 1
print(current_num * current_string)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](image-2.png)




### 160.相交链表

linked list, https://leetcode.cn/problems/intersection-of-two-linked-lists/



思路：
直接做。


代码：

```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        s = set()
        p, q = headA, headB
        while p:
            s.add(p)
            p = p.next
        while q:
            if q in s: return q
            q = q.next
        return None     
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](image-3.png)




### 206.反转链表

linked list, https://leetcode.cn/problems/reverse-linked-list/



思路：
试了半天才知道这题想让我提交什么（反转链表的表头）。用链表中每个点指向上一个点即可，每次都记录原链表的上一个点即可。


代码：

```python
class Solution(object):
    def reverseList(self, head):
        pre = None
        q = head
        while q:
            tmp = q.next
            q.next = pre; pre = q
            q = tmp
        return pre
        
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](image-4.png)




### 3478.选出和最大的K个元素

heap, https://leetcode.cn/problems/choose-k-elements-with-maximum-sum/



思路：
排序后用堆存储最大的k个正值即可。本题坑点在于$nums2[j]<nums2[i]$,故要在循环前记录上一次的current_sum作为当前值的答案。


代码：

```python
class Solution(object):
    def findMaxSum(self, nums1, nums2, k):
        n = len(nums1)
        a = sorted(zip(nums1, nums2, range(n)), key = lambda x: x[0])
        q_large_k = []; q_large_k_len, cur_sum, ans = 0, 0, [0] * n 
        index = 0
        while index < n:
            last = index
            beforehand = cur_sum
            while index < n and a[index][0] == a[last][0]: 
                x = a[index][1]
                if x > 0: 
                    if q_large_k_len < k:
                        heapq.heappush(q_large_k, x)
                        q_large_k_len += 1
                        cur_sum += x
                    elif x > q_large_k[0]:
                        delete = heapq.heappop(q_large_k)
                        heapq.heappush(q_large_k, x)
                        cur_sum += x - delete
                index += 1
            for j in range(last, index): ans[a[j][2]] = beforehand
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![alt text](image-5.png)




### Q6.交互可视化neural network

https://developers.google.com/machine-learning/crash-course/neural-networks/interactive-exercises

**Your task:** configure a neural network that can separate the orange dots from the blue dots in the diagram, achieving a loss of less than 0.2 on both the training and test data.

**Instructions:**

In the interactive widget:

1. Modify the neural network hyperparameters by experimenting with some of the following config settings:
   - Add or remove hidden layers by clicking the **+** and **-** buttons to the left of the **HIDDEN LAYERS** heading in the network diagram.
   - Add or remove neurons from a hidden layer by clicking the **+** and **-** buttons above a hidden-layer column.
   - Change the learning rate by choosing a new value from the **Learning rate** drop-down above the diagram.
   - Change the activation function by choosing a new value from the **Activation** drop-down above the diagram.
2. Click the Play button above the diagram to train the neural network model using the specified parameters.
3. Observe the visualization of the model fitting the data as training progresses, as well as the **Test loss** and **Training loss** values in the **Output** section.
4. If the model does not achieve loss below 0.2 on the test and training data, click reset, and repeat steps 1–3 with a different set of configuration settings. Repeat this process until you achieve the preferred results.

给出满足约束条件的<mark>截图</mark>，并说明学习到的概念和原理。

![alt text](image.png)

感觉挺有意思，再增加隐藏层数有可能出现会过拟合等问题，测试损失反而增大。


## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

本周做作业花了较长时间，之后要多练习、提高熟练度。

数算2025spring每日选做全部AC。








