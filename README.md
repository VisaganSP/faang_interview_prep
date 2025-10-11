# 🎯 FAANG Complete Interview Guide - 18 Month Roadmap
#### Visagan S
*Optimized for 1 hour daily + 2-3 hours weekends*

## 📊 Your Schedule Overview
- **Weekdays**: 1 hour/day (5 hours/week)
- **Weekends**: 2-3 hours/day (4-6 hours/week)
- **Total**: ~11 hours/week
- **Timeline**: 18 months to be fully prepared
- **Total Problems Target**: 400-500 problems

---

## 🗺️ Complete FAANG Interview Components

### What FAANG Companies Actually Test:

1. **📝 Online Assessment (OA)**
   - 2-3 coding problems in 60-90 minutes
   - Usually LeetCode Medium level
   - Some companies include debugging questions

2. **💻 Technical Rounds (2-4 rounds)**
   - Data Structures & Algorithms
   - Problem-solving in 45 minutes
   - Code quality and optimization
   - Testing and edge cases

3. **🏗️ System Design (Senior roles, sometimes for SDE1)**
   - Design scalable systems
   - 45-60 minute discussion
   - Whiteboard architecture

4. **🎭 Behavioral Interview**
   - Leadership principles (Amazon)
   - Googleyness (Google)
   - Cultural fit
   - Past experiences using STAR format

5. **🎯 Domain-Specific (Sometimes)**
   - Mobile development (iOS/Android)
   - Frontend (React/Angular)
   - ML/AI rounds
   - Security rounds

---

## 📅 Phase-by-Phase Breakdown

## **Phase 1: Language Mastery & Basics (Months 1-3)**
*Foundation building - Don't skip this!*

### Month 1: Advanced Java Concepts

#### Week 1-2: Core Java Deep Dive
**Weekday (1hr)**: Watch videos + take notes
**Weekend (2-3hrs)**: Implement + practice

**📹 Video Resources:**
- [Java Programming Tutorial Full Course](https://www.youtube.com/watch?v=eIrMbAQSU34) - Java basics recap
- [Advanced Java Programming](https://www.youtube.com/playlist?list=PLkeaG1zpPTHiMjczpmZ6ALd46VjjiQJ_8) - Advanced concepts
- [Java Multithreading](https://www.youtube.com/watch?v=r_MbozD32eo) - Defog Tech
- [Java Memory Management](https://www.youtube.com/watch?v=LTnp79Ke8FI) - Tech Dummies

**Topics Checklist:**
- [ ] Generics and Type Erasure
- [ ] Collections Framework internals
- [ ] HashMap internal working (MUST KNOW!)
- [ ] ConcurrentHashMap vs HashMap
- [ ] Multithreading - Thread, Runnable, Callable
- [ ] Synchronized, volatile, atomic
- [ ] Thread pools and Executors
- [ ] Deadlock, race conditions

**Code Practice:**
```java
// Master these Java-specific implementations:

// 1. Custom Thread-safe Singleton
public class Singleton {
    private static volatile Singleton instance;
    private Singleton() {}
    
    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}

// 2. Producer-Consumer Problem
class ProducerConsumer {
    Queue<Integer> queue = new LinkedList<>();
    int capacity = 5;
    
    public synchronized void produce() throws InterruptedException {
        while (queue.size() == capacity) wait();
        queue.offer(1);
        notifyAll();
    }
    
    public synchronized void consume() throws InterruptedException {
        while (queue.isEmpty()) wait();
        queue.poll();
        notifyAll();
    }
}

// 3. LRU Cache Implementation
class LRUCache {
    class Node {
        int key, value;
        Node prev, next;
    }
    
    Map<Integer, Node> map;
    Node head, tail;
    int capacity;
    // ... implementation
}
```

#### Week 3-4: Advanced Python Concepts

**📹 Video Resources:**
- [Complete Python Course](https://www.youtube.com/watch?v=_uQrJ0TkZlc) - Programming with Mosh
- [Python Advanced Topics](https://www.youtube.com/watch?v=HGOBQPFzWKo) - FreeCodeCamp
- [Python Decorators](https://www.youtube.com/watch?v=MYAEv3JoenI) - Tech With Tim
- [Python Memory Management](https://www.youtube.com/watch?v=arxWaw-E8QQ) - PyCon

**Topics Checklist:**
- [ ] Decorators and property decorators
- [ ] Generators vs Iterators
- [ ] Context managers (with statement)
- [ ] *args and **kwargs
- [ ] List/Dict/Set comprehensions
- [ ] Lambda functions
- [ ] Global Interpreter Lock (GIL)
- [ ] Async/await basics

**Code Practice:**
```python
# Master these Python patterns:

# 1. Custom Decorator
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Took {time.time() - start}s")
        return result
    return wrapper

# 2. Context Manager
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
        
    def __exit__(self, *args):
        self.file.close()

# 3. Generator for Memory Efficiency
def fibonacci_gen(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 4. Python-specific collections
from collections import defaultdict, Counter, deque
from heapq import heappush, heappop
from bisect import bisect_left, bisect_right
```

### Month 2: Data Structures Implementation

#### Week 1-2: Arrays, Strings, Linked Lists

**📹 Video Resources:**
- [Data Structures Full Course](https://www.youtube.com/watch?v=RBSGKlAvoiM) - FreeCodeCamp
- [Arrays Complete Tutorial](https://www.youtube.com/watch?v=MdvzlDIdQ0o) - CS Dojo
- [Linked List Masterclass](https://www.youtube.com/watch?v=Hj_rA0dhr2I) - FreeCodeCamp

**Weekday Practice Plan (1 hour):**
- Day 1: Watch video on concept (30 min) + Implement basic operations (30 min)
- Day 2: Solve 2 Easy problems
- Day 3: Solve 1 Easy + 1 Medium problem
- Day 4: Watch pattern video + notes
- Day 5: Implement one variation

**Weekend Practice (2-3 hours):**
- Saturday: Implement complete data structure from scratch
- Sunday: Solve 5-6 problems + review solutions

**Must-Implement List:**
```python
# 1. Dynamic Array
class DynamicArray:
    def __init__(self):
        self.capacity = 2
        self.size = 0
        self.arr = [0] * self.capacity
    
    def append(self, val):
        if self.size == self.capacity:
            self._resize()
        self.arr[self.size] = val
        self.size += 1
    
    def _resize(self):
        self.capacity *= 2
        new_arr = [0] * self.capacity
        for i in range(self.size):
            new_arr[i] = self.arr[i]
        self.arr = new_arr

# 2. All Linked List Variants
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Implement: reverse, merge, detect cycle, find middle
```

#### Week 3-4: Stacks, Queues, Hash Tables

**📹 Video Resources:**
- [Stack & Queue Masterclass](https://www.youtube.com/watch?v=A3ZUpyrnCbM) - CS Dojo
- [Hash Table Internals](https://www.youtube.com/watch?v=shs0KM3wKv8) - CS50
- [Design HashMap](https://www.youtube.com/watch?v=cNWsgbKwuNU) - NeetCode

### Month 3: Basic Algorithms & Problem Patterns

#### Week 1-2: Sorting & Searching

**📹 Video Resources:**
- [All Sorting Algorithms](https://www.youtube.com/watch?v=pkkFqlG0Hds) - 15 Sorting Algorithms Visualized
- [Binary Search Deep Dive](https://www.youtube.com/watch?v=P3YID7liBug) - NeetCode
- [Binary Search Patterns](https://www.youtube.com/watch?v=W9QJ8HzRFKk) - Errichto

**Implement All Sorting Algorithms:**
```python
# Quick Sort (MOST ASKED)
def quicksort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)

# Merge Sort (STABLE)
def mergesort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L, R = arr[:mid], arr[mid:]
        mergesort(L)
        mergesort(R)
        # merge logic

# Binary Search Template
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

#### Week 3-4: Two Pointers, Sliding Window, Fast & Slow

**📹 Video Resources:**
- [Two Pointers Technique](https://www.youtube.com/watch?v=On03HWe2tZM) - NeetCode
- [Sliding Window](https://www.youtube.com/watch?v=GcW4mgmgSbw) - Aditya Verma
- [Fast & Slow Pointers](https://www.youtube.com/watch?v=gBTe7lFR3vc) - NeetCode

---

## **Phase 2: Core DSA Mastery (Months 4-8)**

### Month 4-5: Trees & Binary Search Trees

#### Complete Tree Curriculum

**📹 Must-Watch Video Series:**
- [Binary Tree Complete Series](https://www.youtube.com/playlist?list=PLgUwDviBIf0q8Hkd7bK2Bpryj2xVJk8Vk) - TakeUForward (Striver)
- [Binary Search Tree](https://www.youtube.com/watch?v=COZK7NATh4k) - mycodeschool
- [Tree Traversals - All Methods](https://www.youtube.com/watch?v=98AGQU0z2wg) - Back to Back SWE

**Weekday Focus (1 hour):**
- Mon: Learn one tree concept
- Tue: Implement the concept
- Wed: Solve related easy problem
- Thu: Watch problem-solving video
- Fri: Solve medium problem

**Weekend Deep Dive (2-3 hours):**
- Saturday: Implement 2 tree algorithms from scratch
- Sunday: Solve 4-5 tree problems

**Master These Tree Patterns:**
```python
# 1. All Traversals (Recursive + Iterative)
def inorder_iterative(root):
    result, stack = [], []
    current = root
    while stack or current:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    return result

# 2. Level Order Traversal (BFS)
def levelOrder(root):
    if not root: return []
    result, queue = [], [root]
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result

# 3. Tree Construction
def buildTree(preorder, inorder):
    # Construct tree from traversals
    pass

# 4. LCA Pattern
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    return left if left else right
```

### Month 6-7: Graphs Complete

#### Graph Mastery Path

**📹 Complete Graph Playlist:**
- [Graph Series by Striver](https://www.youtube.com/playlist?list=PLgUwDviBIf0oE3gA41TKO2H5bHpPd7fzn) - 54 videos!
- [Graph Algorithms by William Fiset](https://www.youtube.com/playlist?list=PLDV1Zeh2NRsDGO4--qE8yH72HFL1Km93P)
- [Dijkstra's Algorithm](https://www.youtube.com/watch?v=pSqmAO-m7Lk) - Computerphile

**Critical Graph Patterns:**
```python
# 1. BFS Template
def bfs(graph, start):
    visited = set()
    queue = [start]
    visited.add(start)
    
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# 2. DFS Template
def dfs(graph, node, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# 3. Dijkstra's Algorithm
import heapq
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        curr_dist, curr = heapq.heappop(pq)
        if curr_dist > distances[curr]:
            continue
        for neighbor, weight in graph[curr].items():
            distance = curr_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

# 4. Union Find Template
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

### Month 8: Dynamic Programming

#### DP Learning Path

**📹 Best DP Resources:**
- [Dynamic Programming Playlist](https://www.youtube.com/playlist?list=PLgUwDviBIf0qUlt5H_kiKYaNSqJ81PMMY) - Striver (Best!)
- [DP on Trees](https://www.youtube.com/watch?v=gm4Ye0fESpU) - Algorithms Live
- [DP Patterns](https://www.youtube.com/watch?v=oBt53YbR9Kk) - FreeCodeCamp (5 hours)

**DP Problem Categories:**
1. **Linear DP**: Fibonacci, House Robber, Climbing Stairs
2. **Grid DP**: Unique Paths, Min Path Sum
3. **String DP**: LCS, Edit Distance, Palindromes
4. **Decision Making**: Knapsack, Buy/Sell Stock
5. **Partition DP**: MCM, Burst Balloons
6. **Game Theory DP**: Stone Game, Predict Winner

**DP Template to Master:**
```python
# 1. Top-Down (Memoization)
def solve(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = solve(n-1, memo) + solve(n-2, memo)
    return memo[n]

# 2. Bottom-Up (Tabulation)
def solve(n):
    if n <= 1: return n
    dp = [0] * (n + 1)
    dp[0], dp[1] = 0, 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# 3. Space Optimized
def solve(n):
    if n <= 1: return n
    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr
    return prev1
```

---

## **Phase 3: System Design (Months 9-11)**

### Month 9: System Design Fundamentals

**📹 Complete Video Course:**
- [System Design Primer Course](https://www.youtube.com/watch?v=FSR1s2b-l_I&list=PLTCrU9sGyburBw9wNOHebv9SjlE4Elv5a) - sudoCODE
- [Gaurav Sen System Design](https://www.youtube.com/playlist?list=PLMCXHnjXnTnvo6alSjVkgxV-VH6EPyvoX) - Complete playlist
- [System Design Interview](https://www.youtube.com/c/SystemDesignInterview) - All topics

**Week-by-Week Topics:**

**Week 1: Networking & Basics**
- [OSI Model Explained](https://www.youtube.com/watch?v=vv4y_uOneC0)
- [TCP vs UDP](https://www.youtube.com/watch?v=uwoD5YsGACg)
- [HTTP/HTTPS](https://www.youtube.com/watch?v=hExRDVZHhig)
- [REST vs GraphQL](https://www.youtube.com/watch?v=yWzKJPw_VzM)

**Week 2: Databases**
- [SQL vs NoSQL](https://www.youtube.com/watch?v=ZS_kXvOeQ5Y)
- [ACID Properties](https://www.youtube.com/watch?v=pomDRN3SISE)
- [CAP Theorem](https://www.youtube.com/watch?v=k-Yaq8AHlFA)
- [Database Sharding](https://www.youtube.com/watch?v=hdxdhCpgYo8)
- [Database Indexing](https://www.youtube.com/watch?v=3G293is403I)

**Week 3: Caching & CDN**
- [Caching Strategies](https://www.youtube.com/watch?v=U3RkDLtS7uY)
- [Redis Explained](https://www.youtube.com/watch?v=rP9EKvWt0zo)
- [CDN Deep Dive](https://www.youtube.com/watch?v=Bsq5cKkS33I)

**Week 4: Message Queues & Microservices**
- [Message Queue Systems](https://www.youtube.com/watch?v=oUJbuFMyBDk)
- [Kafka Architecture](https://www.youtube.com/watch?v=aj9CDyypKSA)
- [Microservices](https://www.youtube.com/watch?v=j1gU2oGFayY)

### Month 10-11: System Design Practice

**📹 Design These Systems (with video walkthroughs):**

1. **URL Shortener**
   - [Video Solution](https://www.youtube.com/watch?v=JQDHz72OA3c)
   - Key: Base62 encoding, Custom URLs, Analytics

2. **WhatsApp/Messenger**
   - [Video Solution](https://www.youtube.com/watch?v=vvhC64hQZMk)
   - Key: WebSockets, Message Queue, Last seen

3. **Instagram**
   - [Video Solution](https://www.youtube.com/watch?v=QmX2NPkJyPk)
   - Key: Timeline generation, Image storage, News feed

4. **Uber/Lyft**
   - [Video Solution](https://www.youtube.com/watch?v=umWABit-wbk)
   - Key: Geospatial index, Dispatch system, Pricing

5. **YouTube/Netflix**
   - [Video Solution](https://www.youtube.com/watch?v=psQzyFfsUGU)
   - Key: Video streaming, CDN, Recommendation

6. **Twitter**
   - [Video Solution](https://www.youtube.com/watch?v=wYk0xPP_P_8)
   - Key: Timeline generation, Celebrity tweets, Trending

7. **Google Docs**
   - [Video Solution](https://www.youtube.com/watch?v=2auwirNBvGg)
   - Key: Operational Transform, Conflict resolution

8. **Distributed Cache**
   - [Video Solution](https://www.youtube.com/watch?v=iuqZvajTOyA)
   - Key: Consistent hashing, Replication

9. **Rate Limiter**
   - [Video Solution](https://www.youtube.com/watch?v=FU4WlwfS3G0)
   - Key: Token bucket, Sliding window

10. **Payment System**
    - [Video Solution](https://www.youtube.com/watch?v=olfaBgJrUBI)
    - Key: Idempotency, Transaction, Security

**System Design Template:**
```
1. Requirements Clarification (5 min)
   - Functional requirements
   - Non-functional requirements
   - Scale estimation

2. Capacity Estimation (5 min)
   - Storage requirements
   - Bandwidth requirements
   - Memory requirements

3. API Design (5 min)
   - REST endpoints
   - Input/Output format

4. Data Model (5 min)
   - Schema design
   - Database choice

5. High-Level Design (10 min)
   - Draw basic components
   - Show data flow

6. Detailed Design (15 min)
   - Deep dive into components
   - Algorithm choices

7. Scale & Optimize (10 min)
   - Identify bottlenecks
   - Add caching, CDN, etc.

8. Handle Failures (5 min)
   - Single points of failure
   - Monitoring & alerts
```

---

## **Phase 4: Advanced Algorithms (Months 12-14)**

### Month 12: Heap, Trie, Advanced DS

**📹 Video Resources:**
- [Heap/Priority Queue Complete](https://www.youtube.com/watch?v=HqPJF2L5h9U) - Abdul Bari
- [Trie Data Structure](https://www.youtube.com/watch?v=AXjmTQ8LEoI) - NeetCode
- [Segment Tree](https://www.youtube.com/watch?v=2FShdqn-Oz8) - Tushar Roy

### Month 13: Bit Manipulation & Math

**📹 Resources:**
- [Bit Manipulation Complete](https://www.youtube.com/watch?v=5rtVTYAk9KQ)
- [Number Theory for CP](https://www.youtube.com/watch?v=JBG6r5KNYVY)

**Must-Know Bit Tricks:**
```python
# 1. Check if power of 2
def isPowerOfTwo(n):
    return n > 0 and (n & (n - 1)) == 0

# 2. Count set bits
def countBits(n):
    count = 0
    while n:
        n &= (n - 1)  # Remove rightmost set bit
        count += 1
    return count

# 3. XOR properties
# a ^ a = 0
# a ^ 0 = a
# a ^ b ^ a = b
```

### Month 14: Greedy & Backtracking

**📹 Resources:**
- [Greedy Algorithms](https://www.youtube.com/watch?v=ARvQcqJ_-NY)
- [Backtracking](https://www.youtube.com/watch?v=A80YzvNwqXA) - N Queens

---

## **Phase 5: Interview Preparation (Months 15-18)**

### Month 15-16: Company-Specific Prep

#### Amazon Specific

**📹 Amazon Prep:**
- [Amazon Leadership Principles](https://www.youtube.com/watch?v=dse8OTDlRcE)
- [Amazon Bar Raiser](https://www.youtube.com/watch?v=B-xdfnKGdCY)

**Amazon's 16 Leadership Principles - Prepare 2 Stories Each:**
1. Customer Obsession
2. Ownership
3. Invent and Simplify
4. Are Right, A Lot
5. Learn and Be Curious
6. Hire and Develop the Best
7. Insist on the Highest Standards
8. Think Big
9. Bias for Action
10. Frugality
11. Earn Trust
12. Dive Deep
13. Have Backbone; Disagree and Commit
14. Deliver Results
15. Strive to be Earth's Best Employer
16. Success and Scale Bring Broad Responsibility

**Amazon OA Pattern:**
- 2 coding problems (70 minutes)
- Debugging (20 minutes)
- Work simulation
- Logical reasoning

#### Google Specific

**📹 Google Prep:**
- [Google Interview Process](https://www.youtube.com/watch?v=XKu_SEDAykw)
- [Googliness](https://www.youtube.com/watch?v=3Q_oYDQ2whs)

**Google Focus Areas:**
- Algorithm optimization
- Code quality
- Testing (write test cases!)
- Scalability discussions

#### Meta (Facebook) Specific

**📹 Meta Prep:**
- [Meta Interview Experience](https://www.youtube.com/watch?v=bHK4cjB7l6M)
- [Facebook System Design](https://www.youtube.com/watch?v=lbaPoh8fhNI)

**Meta Focus:**
- Speed of coding
- Move fast and break things
- Practical implementation

#### Microsoft Specific

**📹 Microsoft Prep:**
- [Microsoft Interview](https://www.youtube.com/watch?v=VPHKr2Axqas)

**Microsoft Focus:**
- Growth mindset
- Inclusive design
- Azure/Cloud questions

### Month 17: Mock Interviews

**Mock Interview Platforms:**

1. **Free Platforms:**
   - [Pramp](https://www.pramp.com) - Peer mock interviews
   - [Interviewing.io](https://interviewing.io) - Anonymous practice

2. **Paid Platforms:**
   - [Exponent](https://www.tryexponent.com) - $12/month
   - [Interview Kickstart](https://www.interviewkickstart.com)
   - [Pramp Pro](https://www.pramp.com) - Professional interviewers

**Self-Mock Interview Schedule:**
- Week 1-2: 2 coding mocks/week
- Week 3-4: 1 system design mock/week
- Throughout: 1 behavioral mock/week

### Month 18: Final Polish

**Last Month Checklist:**

**Week 1-2: Weak Areas**
- Identify top 3 weak topics
- Solve 50 problems in weak areas
- Watch explanations for each

**Week 3: Company Research**
- Read recent tech blogs
- Understand products
- Know the tech stack
- Prepare questions to ask

**Week 4: Mental Preparation**
- Review all notes
- Light practice only
- Mock interviews
- Rest well!

---

## **Behavioral Interview Complete Guide**

### STAR Method Framework

**S**ituation - Context/background
**T**ask - Challenge/responsibility
**A**ction - What YOU did
**R**esult - Outcome/learning

### Must-Have Stories (Prepare 2-3 Each):

1. **Leadership Story**
   - Led a team/project
   - Influenced without authority
   - Mentored someone

2. **Conflict Story**
   - Disagreement with teammate
   - Handled difficult person
   - Competing priorities

3. **Failure Story**
   - Made a mistake
   - Missed deadline
   - Project failed

4. **Success Story**
   - Biggest achievement
   - Exceeded expectations
   - Innovation/creativity

5. **Challenge Story**
   - Hardest problem solved
   - Ambiguous situation
   - Technical complexity

**📹 Behavioral Prep Videos:**
- [STAR Method Explained](https://www.youtube.com/watch?v=5v0pJ5hDd0k)
- [Amazon Behavioral](https://www.youtube.com/watch?v=dse8OTDlRcE)
- [Google Behavioral](https://www.youtube.com/watch?v=0hSjcYnWGiI)

---

## **Your Weekly Schedule Template**

### Weekday (1 Hour)
**Option A - Problem Solving Day:**
- 10 min: Review yesterday's solution
- 40 min: Solve 1 new problem
- 10 min: Study optimal solution

**Option B - Learning Day:**
- 30 min: Watch concept video
- 30 min: Implement/take notes

### Saturday (2-3 Hours)
**Morning (1.5 hrs):**
- Implement one DS/Algorithm from scratch
- Solve 2 medium problems

**Afternoon (1.5 hrs):**
- System design study/practice
- OR Mock interview

### Sunday (2-3 Hours)
**Morning (1.5 hrs):**
- Weekly contest (LeetCode/Codeforces)
- Review solutions

**Afternoon (1.5 hrs):**
- Solve 3-4 problems
- Behavioral story writing

---

## **Problem-Solving Resources**

### Platforms Priority Order:
1. **LeetCode** (Primary - Get Premium!)
   - Sort by company tags
   - Use frequency sorting
   - Study top discussions

2. **NeetCode.io** (Free!)
   - Curated problem lists
   - Video explanations
   - Roadmap structure

3. **AlgoExpert** ($99/year)
   - 160 curated problems
   - Video explanations
   - Certificate

### YouTube Channels (Subscribe All):
1. **NeetCode** - Best explanations
2. **Take U Forward (Striver)** - Complete DSA
3. **Back to Back SWE** - Whiteboard style
4. **Tushar Roy** - DP specialist
5. **Gaurav Sen** - System design
6. **Clement Mihailescu** - AlgoExpert CEO
7. **Kevin Naughton Jr** - LeetCode solutions
8. **Nick White** - Live coding
9. **Errichto** - Competitive programming
10. **William Fiset** - Graph algorithms

### Books (PDF/Physical):
1. **Cracking the Coding Interview** - Gayle McDowell
2. **Elements of Programming Interviews** (EPI)
3. **System Design Interview** - Alex Xu (Vol 1 & 2)
4. **Designing Data-Intensive Applications** - Kleppmann

### GitHub Resources:
- [Awesome Interview Questions](https://github.com/DopplerHQ/awesome-interview-questions)
- [System Design Primer](https://github.com/donnemartin/system-design-primer)
- [LeetCode Patterns](https://github.com/seanprashad/leetcode-patterns)

---

## **Interview Day Strategy**

### Before Interview:
- [ ] Test equipment (camera, mic, internet)
- [ ] Prepare water bottle
- [ ] Have pen and paper ready
- [ ] Open IDE for reference (but don't use)
- [ ] Review STAR stories
- [ ] Do 1 easy warm-up problem

### During Coding Interview:

**First 5-10 Minutes:**
1. Clarify requirements
2. Ask about edge cases
3. Discuss input constraints
4. Confirm output format

**Next 5 Minutes:**
1. Discuss approach
2. Consider multiple solutions
3. Analyze complexity
4. Get interviewer buy-in

**Next 25 Minutes:**
1. Code the solution
2. Talk while coding
3. Use good variable names
4. Write clean code

**Last 5-10 Minutes:**
1. Test with examples
2. Handle edge cases
3. Discuss optimizations
4. Ask questions

### Common Mistakes to Avoid:
❌ Starting coding immediately
❌ Not asking clarifying questions
❌ Silent coding
❌ Ignoring hints
❌ Getting defensive
❌ Not testing code
❌ Poor variable names
❌ Not considering edge cases

---

## **Tracking Progress Spreadsheet**

Create a spreadsheet with these columns:
- Date
- Problem Name
- Platform
- Difficulty
- Category (Array/Tree/DP etc)
- Time Taken
- Solved (Yes/No/Partial)
- Need Review (Yes/No)
- Notes
- Company Tags

**Weekly Metrics:**
- Problems solved
- Success rate
- Average time
- Weak areas identified

---

## **Motivation & Mental Health**

### When You Feel Stuck:
1. Take a break (go for walk)
2. Review basics again
3. Watch solution video
4. Discuss with peers
5. Post in forums

### Dealing with Rejection:
- It's normal (even experts get rejected)
- Learn from feedback
- Keep practicing
- Apply to more companies
- Remember: It's a numbers game

### Success Stories for Motivation:
- Search "FAANG interview experience" on:
  - Reddit (r/cscareerquestions)
  - Blind App
  - LeetCode Discuss
  - YouTube

---

## **Final Pro Tips**

### Language Choice:
- **Use Python** if you want to code faster
- **Use Java** if interviewing at Amazon/Microsoft
- **Stick to one language** during prep

### During Preparation:
1. **Quality > Quantity**: Understanding 200 problems > rushing 500
2. **Review is key**: Revisit problems after 1 week, 1 month
3. **Pattern Recognition**: Group similar problems
4. **Time yourself**: Always practice with timer
5. **Write clean code**: Even during practice

### Red Flags to Avoid:
- Don't memorize solutions
- Don't ignore behavioral prep
- Don't apply too early
- Don't ignore system design (even for SDE1)
- Don't compare with others

### Green Flags (You're Ready):
✅ Solve medium problems in 25-30 min consistently
✅ Can explain your thought process clearly
✅ Recognize patterns immediately
✅ Have 10+ behavioral stories ready
✅ Can design basic systems
✅ Feel confident, not perfect

---

## **Company Application Strategy**

### Application Timeline:
1. **Month 15**: Start applying to dream companies
2. **Month 16**: Apply to target companies
3. **Month 17**: Apply to backup companies
4. **Month 18**: Focus on scheduled interviews

### Where to Apply:
- Company careers page (best)
- LinkedIn (good response rate)
- AngelList (startups)
- Referrals (highest success rate!)

### Getting Referrals:
1. LinkedIn - Connect with employees
2. Blind App - Ask for referrals
3. College alumni network
4. Meetups and conferences
5. Open source contributions

---

## **Remember: You Can Do This! 💪**

This journey will be challenging but transformative. Every problem you solve makes you a better engineer. Trust the process, stay consistent with your 1 hour daily + weekend schedule, and you'll be ready for FAANG in 18 months!

**Your Success Formula:**
```
Consistency + Pattern Recognition + Communication Skills = FAANG Offer
```

Good luck! Start today, not tomorrow! 🚀

---

*Document Version: 2.0*
*Optimized for: 11 hours/week schedule*
*Timeline: 18 months*
*Last Updated: October 2024*