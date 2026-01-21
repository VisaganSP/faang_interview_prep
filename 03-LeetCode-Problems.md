# FAANG LeetCode Problems - Curated Interview Questions

## How to Use This Guide
Problems are organized by topic. Each problem includes optimal solutions in Java and Python with complexity analysis.

**Difficulty:** 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## 1. Arrays & Hashing

### 🟢 Two Sum (#1)
**Java:**
```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (map.containsKey(complement)) {
            return new int[]{map.get(complement), i};
        }
        map.put(nums[i], i);
    }
    return new int[]{};
}
```
**Python:**
```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
    return []
```
**Complexity:** Time O(n), Space O(n)

---

### 🟡 Product of Array Except Self (#238)
**Java:**
```java
public int[] productExceptSelf(int[] nums) {
    int n = nums.length;
    int[] result = new int[n];
    result[0] = 1;
    for (int i = 1; i < n; i++) result[i] = result[i-1] * nums[i-1];
    int suffix = 1;
    for (int i = n - 1; i >= 0; i--) {
        result[i] *= suffix;
        suffix *= nums[i];
    }
    return result;
}
```
**Python:**
```python
def productExceptSelf(nums):
    n = len(nums)
    result = [1] * n
    for i in range(1, n): result[i] = result[i-1] * nums[i-1]
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    return result
```
**Complexity:** Time O(n), Space O(1)

---

### 🟡 Longest Consecutive Sequence (#128)
**Java:**
```java
public int longestConsecutive(int[] nums) {
    Set<Integer> set = new HashSet<>();
    for (int num : nums) set.add(num);
    int maxLen = 0;
    for (int num : set) {
        if (!set.contains(num - 1)) {
            int len = 1;
            while (set.contains(num + len)) len++;
            maxLen = Math.max(maxLen, len);
        }
    }
    return maxLen;
}
```
**Python:**
```python
def longestConsecutive(nums):
    num_set = set(nums)
    max_len = 0
    for num in num_set:
        if num - 1 not in num_set:
            length = 1
            while num + length in num_set: length += 1
            max_len = max(max_len, length)
    return max_len
```
**Complexity:** Time O(n), Space O(n)

---

## 2. Two Pointers

### 🟡 3Sum (#15)
**Java:**
```java
public List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    Arrays.sort(nums);
    for (int i = 0; i < nums.length - 2; i++) {
        if (i > 0 && nums[i] == nums[i-1]) continue;
        int left = i + 1, right = nums.length - 1;
        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];
            if (sum == 0) {
                result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                while (left < right && nums[left] == nums[left+1]) left++;
                while (left < right && nums[right] == nums[right-1]) right--;
                left++; right--;
            } else if (sum < 0) left++;
            else right--;
        }
    }
    return result;
}
```
**Python:**
```python
def threeSum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]: continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]: left += 1
                while left < right and nums[right] == nums[right-1]: right -= 1
                left += 1; right -= 1
            elif total < 0: left += 1
            else: right -= 1
    return result
```
**Complexity:** Time O(n²), Space O(1)

---

### 🔴 Trapping Rain Water (#42)
**Java:**
```java
public int trap(int[] height) {
    int left = 0, right = height.length - 1;
    int leftMax = 0, rightMax = 0, water = 0;
    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) leftMax = height[left];
            else water += leftMax - height[left];
            left++;
        } else {
            if (height[right] >= rightMax) rightMax = height[right];
            else water += rightMax - height[right];
            right--;
        }
    }
    return water;
}
```
**Python:**
```python
def trap(height):
    left, right = 0, len(height) - 1
    left_max = right_max = water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max: left_max = height[left]
            else: water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max: right_max = height[right]
            else: water += right_max - height[right]
            right -= 1
    return water
```
**Complexity:** Time O(n), Space O(1)

---

## 3. Sliding Window

### 🟡 Longest Substring Without Repeating Characters (#3)
**Java:**
```java
public int lengthOfLongestSubstring(String s) {
    Map<Character, Integer> map = new HashMap<>();
    int left = 0, maxLen = 0;
    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        if (map.containsKey(c)) left = Math.max(left, map.get(c) + 1);
        map.put(c, right);
        maxLen = Math.max(maxLen, right - left + 1);
    }
    return maxLen;
}
```
**Python:**
```python
def lengthOfLongestSubstring(s):
    char_idx = {}
    left = max_len = 0
    for right, char in enumerate(s):
        if char in char_idx and char_idx[char] >= left:
            left = char_idx[char] + 1
        char_idx[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len
```
**Complexity:** Time O(n), Space O(min(n, 26))

---

### 🔴 Minimum Window Substring (#76)
**Java:**
```java
public String minWindow(String s, String t) {
    Map<Character, Integer> need = new HashMap<>(), window = new HashMap<>();
    for (char c : t.toCharArray()) need.put(c, need.getOrDefault(c, 0) + 1);
    int left = 0, minStart = 0, minLen = Integer.MAX_VALUE, formed = 0;
    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        window.put(c, window.getOrDefault(c, 0) + 1);
        if (need.containsKey(c) && window.get(c).equals(need.get(c))) formed++;
        while (formed == need.size()) {
            if (right - left + 1 < minLen) { minLen = right - left + 1; minStart = left; }
            char leftChar = s.charAt(left);
            window.put(leftChar, window.get(leftChar) - 1);
            if (need.containsKey(leftChar) && window.get(leftChar) < need.get(leftChar)) formed--;
            left++;
        }
    }
    return minLen == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLen);
}
```
**Python:**
```python
from collections import Counter
def minWindow(s, t):
    need, window = Counter(t), {}
    left, min_start, min_len, formed = 0, 0, float('inf'), 0
    for right, char in enumerate(s):
        window[char] = window.get(char, 0) + 1
        if char in need and window[char] == need[char]: formed += 1
        while formed == len(need):
            if right - left + 1 < min_len: min_len, min_start = right - left + 1, left
            window[s[left]] -= 1
            if s[left] in need and window[s[left]] < need[s[left]]: formed -= 1
            left += 1
    return "" if min_len == float('inf') else s[min_start:min_start + min_len]
```
**Complexity:** Time O(n + m), Space O(n + m)

---

## 4. Binary Search

### 🟡 Search in Rotated Sorted Array (#33)
**Java:**
```java
public int search(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[left] <= nums[mid]) {
            if (target >= nums[left] && target < nums[mid]) right = mid - 1;
            else left = mid + 1;
        } else {
            if (target > nums[mid] && target <= nums[right]) left = mid + 1;
            else right = mid - 1;
        }
    }
    return -1;
}
```
**Python:**
```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target: return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]: right = mid - 1
            else: left = mid + 1
        else:
            if nums[mid] < target <= nums[right]: left = mid + 1
            else: right = mid - 1
    return -1
```
**Complexity:** Time O(log n), Space O(1)

---

## 5. Linked Lists

### 🟡 Reverse Linked List (#206)
**Java:**
```java
public ListNode reverseList(ListNode head) {
    ListNode prev = null, curr = head;
    while (curr != null) {
        ListNode next = curr.next;
        curr.next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}
```
**Python:**
```python
def reverseList(head):
    prev, curr = None, head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```
**Complexity:** Time O(n), Space O(1)

---

### 🔴 Merge K Sorted Lists (#23)
**Java:**
```java
public ListNode mergeKLists(ListNode[] lists) {
    PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
    for (ListNode list : lists) if (list != null) pq.offer(list);
    ListNode dummy = new ListNode(0), curr = dummy;
    while (!pq.isEmpty()) {
        ListNode node = pq.poll();
        curr.next = node;
        curr = curr.next;
        if (node.next != null) pq.offer(node.next);
    }
    return dummy.next;
}
```
**Python:**
```python
import heapq
def mergeKLists(lists):
    heap = [(lst.val, i, lst) for i, lst in enumerate(lists) if lst]
    heapq.heapify(heap)
    dummy = curr = ListNode(0)
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next: heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
```
**Complexity:** Time O(n log k), Space O(k)

---

### 🔴 LRU Cache (#146)
**Java:**
```java
class LRUCache {
    private Map<Integer, Node> cache = new HashMap<>();
    private int capacity;
    private Node head = new Node(0, 0), tail = new Node(0, 0);
    class Node { int key, val; Node prev, next; Node(int k, int v) { key=k; val=v; } }
    
    public LRUCache(int capacity) { this.capacity = capacity; head.next = tail; tail.prev = head; }
    
    public int get(int key) {
        if (!cache.containsKey(key)) return -1;
        Node node = cache.get(key); remove(node); insertHead(node);
        return node.val;
    }
    
    public void put(int key, int value) {
        if (cache.containsKey(key)) remove(cache.get(key));
        Node node = new Node(key, value);
        cache.put(key, node); insertHead(node);
        if (cache.size() > capacity) { Node lru = tail.prev; remove(lru); cache.remove(lru.key); }
    }
    
    private void remove(Node n) { n.prev.next = n.next; n.next.prev = n.prev; }
    private void insertHead(Node n) { n.next = head.next; n.prev = head; head.next.prev = n; head.next = n; }
}
```
**Python:**
```python
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity): self.cache, self.capacity = OrderedDict(), capacity
    def get(self, key):
        if key not in self.cache: return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, value):
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity: self.cache.popitem(last=False)
```
**Complexity:** Time O(1), Space O(capacity)

---

## 6. Trees

### 🟡 Validate Binary Search Tree (#98)
**Java:**
```java
public boolean isValidBST(TreeNode root) { return validate(root, Long.MIN_VALUE, Long.MAX_VALUE); }
private boolean validate(TreeNode node, long min, long max) {
    if (node == null) return true;
    if (node.val <= min || node.val >= max) return false;
    return validate(node.left, min, node.val) && validate(node.right, node.val, max);
}
```
**Python:**
```python
def isValidBST(root):
    def validate(node, min_val, max_val):
        if not node: return True
        if node.val <= min_val or node.val >= max_val: return False
        return validate(node.left, min_val, node.val) and validate(node.right, node.val, max_val)
    return validate(root, float('-inf'), float('inf'))
```
**Complexity:** Time O(n), Space O(h)

---

### 🔴 Binary Tree Maximum Path Sum (#124)
**Java:**
```java
private int maxSum = Integer.MIN_VALUE;
public int maxPathSum(TreeNode root) { maxGain(root); return maxSum; }
private int maxGain(TreeNode node) {
    if (node == null) return 0;
    int left = Math.max(maxGain(node.left), 0);
    int right = Math.max(maxGain(node.right), 0);
    maxSum = Math.max(maxSum, node.val + left + right);
    return node.val + Math.max(left, right);
}
```
**Python:**
```python
def maxPathSum(root):
    max_sum = [float('-inf')]
    def maxGain(node):
        if not node: return 0
        left, right = max(maxGain(node.left), 0), max(maxGain(node.right), 0)
        max_sum[0] = max(max_sum[0], node.val + left + right)
        return node.val + max(left, right)
    maxGain(root)
    return max_sum[0]
```
**Complexity:** Time O(n), Space O(h)

---

## 7. Graphs

### 🟡 Number of Islands (#200)
**Java:**
```java
public int numIslands(char[][] grid) {
    int count = 0;
    for (int i = 0; i < grid.length; i++)
        for (int j = 0; j < grid[0].length; j++)
            if (grid[i][j] == '1') { dfs(grid, i, j); count++; }
    return count;
}
private void dfs(char[][] grid, int i, int j) {
    if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != '1') return;
    grid[i][j] = '0';
    dfs(grid, i+1, j); dfs(grid, i-1, j); dfs(grid, i, j+1); dfs(grid, i, j-1);
}
```
**Python:**
```python
def numIslands(grid):
    def dfs(i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != '1': return
        grid[i][j] = '0'
        dfs(i+1, j); dfs(i-1, j); dfs(i, j+1); dfs(i, j-1)
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1': dfs(i, j); count += 1
    return count
```
**Complexity:** Time O(m*n), Space O(m*n)

---

### 🟡 Course Schedule (#207)
**Java:**
```java
public boolean canFinish(int numCourses, int[][] prerequisites) {
    List<List<Integer>> graph = new ArrayList<>();
    int[] inDegree = new int[numCourses];
    for (int i = 0; i < numCourses; i++) graph.add(new ArrayList<>());
    for (int[] pre : prerequisites) { graph.get(pre[1]).add(pre[0]); inDegree[pre[0]]++; }
    Queue<Integer> queue = new LinkedList<>();
    for (int i = 0; i < numCourses; i++) if (inDegree[i] == 0) queue.offer(i);
    int count = 0;
    while (!queue.isEmpty()) {
        int course = queue.poll(); count++;
        for (int next : graph.get(course)) if (--inDegree[next] == 0) queue.offer(next);
    }
    return count == numCourses;
}
```
**Python:**
```python
from collections import defaultdict, deque
def canFinish(numCourses, prerequisites):
    graph, inDegree = defaultdict(list), [0] * numCourses
    for c, p in prerequisites: graph[p].append(c); inDegree[c] += 1
    queue = deque([i for i in range(numCourses) if inDegree[i] == 0])
    count = 0
    while queue:
        course = queue.popleft(); count += 1
        for next_c in graph[course]:
            inDegree[next_c] -= 1
            if inDegree[next_c] == 0: queue.append(next_c)
    return count == numCourses
```
**Complexity:** Time O(V + E), Space O(V + E)

---

## 8. Dynamic Programming

### 🟡 Coin Change (#322)
**Java:**
```java
public int coinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, amount + 1);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++)
        for (int coin : coins)
            if (coin <= i) dp[i] = Math.min(dp[i], dp[i - coin] + 1);
    return dp[amount] > amount ? -1 : dp[amount];
}
```
**Python:**
```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i: dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```
**Complexity:** Time O(amount * n), Space O(amount)

---

### 🟡 Longest Increasing Subsequence (#300)
**Java:**
```java
public int lengthOfLIS(int[] nums) {
    List<Integer> tails = new ArrayList<>();
    for (int num : nums) {
        int pos = Collections.binarySearch(tails, num);
        if (pos < 0) pos = -(pos + 1);
        if (pos == tails.size()) tails.add(num);
        else tails.set(pos, num);
    }
    return tails.size();
}
```
**Python:**
```python
import bisect
def lengthOfLIS(nums):
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails): tails.append(num)
        else: tails[pos] = num
    return len(tails)
```
**Complexity:** Time O(n log n), Space O(n)

---

### 🔴 Edit Distance (#72)
**Java:**
```java
public int minDistance(String word1, String word2) {
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            dp[i][j] = word1.charAt(i-1) == word2.charAt(j-1) ? dp[i-1][j-1] 
                     : 1 + Math.min(dp[i-1][j-1], Math.min(dp[i-1][j], dp[i][j-1]));
    return dp[m][n];
}
```
**Python:**
```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] if word1[i-1] == word2[j-1] else 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```
**Complexity:** Time O(m*n), Space O(m*n)

---

## 9. Backtracking

### 🟡 Subsets (#78)
**Java:**
```java
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(nums, 0, new ArrayList<>(), result);
    return result;
}
private void backtrack(int[] nums, int start, List<Integer> curr, List<List<Integer>> result) {
    result.add(new ArrayList<>(curr));
    for (int i = start; i < nums.length; i++) {
        curr.add(nums[i]);
        backtrack(nums, i + 1, curr, result);
        curr.remove(curr.size() - 1);
    }
}
```
**Python:**
```python
def subsets(nums):
    result = []
    def backtrack(start, curr):
        result.append(curr[:])
        for i in range(start, len(nums)):
            curr.append(nums[i])
            backtrack(i + 1, curr)
            curr.pop()
    backtrack(0, [])
    return result
```
**Complexity:** Time O(n * 2^n), Space O(n)

---

### 🟡 Combination Sum (#39)
**Java:**
```java
public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(candidates, target, 0, new ArrayList<>(), result);
    return result;
}
private void backtrack(int[] cand, int remain, int start, List<Integer> curr, List<List<Integer>> result) {
    if (remain == 0) { result.add(new ArrayList<>(curr)); return; }
    if (remain < 0) return;
    for (int i = start; i < cand.length; i++) {
        curr.add(cand[i]);
        backtrack(cand, remain - cand[i], i, curr, result);
        curr.remove(curr.size() - 1);
    }
}
```
**Python:**
```python
def combinationSum(candidates, target):
    result = []
    def backtrack(remain, start, curr):
        if remain == 0: result.append(curr[:]); return
        if remain < 0: return
        for i in range(start, len(candidates)):
            curr.append(candidates[i])
            backtrack(remain - candidates[i], i, curr)
            curr.pop()
    backtrack(target, 0, [])
    return result
```
**Complexity:** Time O(n^(T/M)), Space O(T/M)

---

## 10. Heap

### 🟡 Kth Largest Element (#215)
**Java:**
```java
public int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    for (int num : nums) {
        minHeap.offer(num);
        if (minHeap.size() > k) minHeap.poll();
    }
    return minHeap.peek();
}
```
**Python:**
```python
import heapq
def findKthLargest(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k: heapq.heappop(heap)
    return heap[0]
```
**Complexity:** Time O(n log k), Space O(k)

---

### 🔴 Find Median from Data Stream (#295)
**Java:**
```java
class MedianFinder {
    PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
    PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    public void addNum(int num) {
        maxHeap.offer(num);
        minHeap.offer(maxHeap.poll());
        if (minHeap.size() > maxHeap.size()) maxHeap.offer(minHeap.poll());
    }
    public double findMedian() {
        return maxHeap.size() > minHeap.size() ? maxHeap.peek() : (maxHeap.peek() + minHeap.peek()) / 2.0;
    }
}
```
**Python:**
```python
import heapq
class MedianFinder:
    def __init__(self): self.maxHeap, self.minHeap = [], []
    def addNum(self, num):
        heapq.heappush(self.maxHeap, -num)
        heapq.heappush(self.minHeap, -heapq.heappop(self.maxHeap))
        if len(self.minHeap) > len(self.maxHeap): heapq.heappush(self.maxHeap, -heapq.heappop(self.minHeap))
    def findMedian(self):
        return -self.maxHeap[0] if len(self.maxHeap) > len(self.minHeap) else (-self.maxHeap[0] + self.minHeap[0]) / 2.0
```
**Complexity:** addNum O(log n), findMedian O(1)

---

## 11. Stack

### 🟢 Valid Parentheses (#20)
**Java:**
```java
public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    for (char c : s.toCharArray()) {
        if (c == '(') stack.push(')');
        else if (c == '{') stack.push('}');
        else if (c == '[') stack.push(']');
        else if (stack.isEmpty() || stack.pop() != c) return false;
    }
    return stack.isEmpty();
}
```
**Python:**
```python
def isValid(s):
    stack, mapping = [], {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in mapping:
            if not stack or stack.pop() != mapping[c]: return False
        else: stack.append(c)
    return len(stack) == 0
```
**Complexity:** Time O(n), Space O(n)

---

### 🔴 Largest Rectangle in Histogram (#84)
**Java:**
```java
public int largestRectangleArea(int[] heights) {
    Stack<Integer> stack = new Stack<>();
    int maxArea = 0;
    for (int i = 0; i <= heights.length; i++) {
        int h = (i == heights.length) ? 0 : heights[i];
        while (!stack.isEmpty() && h < heights[stack.peek()]) {
            int height = heights[stack.pop()];
            int width = stack.isEmpty() ? i : i - stack.peek() - 1;
            maxArea = Math.max(maxArea, height * width);
        }
        stack.push(i);
    }
    return maxArea;
}
```
**Python:**
```python
def largestRectangleArea(heights):
    stack, maxArea = [], 0
    for i in range(len(heights) + 1):
        h = heights[i] if i < len(heights) else 0
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            maxArea = max(maxArea, height * width)
        stack.append(i)
    return maxArea
```
**Complexity:** Time O(n), Space O(n)

---

## Quick Reference - Top 20 FAANG Problems

| # | Problem | Difficulty | Pattern |
|---|---------|------------|---------|
| 1 | Two Sum | Easy | Hash Map |
| 2 | LRU Cache | Hard | Hash + DLL |
| 3 | Merge Intervals | Medium | Sort + Merge |
| 4 | Number of Islands | Medium | DFS/BFS |
| 5 | Reverse Linked List | Easy | Pointers |
| 6 | Valid Parentheses | Easy | Stack |
| 7 | 3Sum | Medium | Two Pointers |
| 8 | Longest Substring No Repeat | Medium | Sliding Window |
| 9 | Search Rotated Array | Medium | Binary Search |
| 10 | Product Except Self | Medium | Prefix/Suffix |
| 11 | Coin Change | Medium | DP |
| 12 | Course Schedule | Medium | Topological Sort |
| 13 | Merge K Sorted Lists | Hard | Heap |
| 14 | Validate BST | Medium | Tree DFS |
| 15 | Min Window Substring | Hard | Sliding Window |
| 16 | Trapping Rain Water | Hard | Two Pointers |
| 17 | Binary Tree Max Path | Hard | Tree DFS |
| 18 | Edit Distance | Hard | DP |
| 19 | LIS | Medium | Binary Search |
| 20 | Largest Rectangle | Hard | Monotonic Stack |

---

## Study Tips
1. **Master patterns** - Most problems fall into ~15 patterns
2. **Understand, don't memorize** - Know why solutions work
3. **Practice under time pressure** - 45 min per problem
4. **Review mistakes** - Keep a log of errors
5. **Communicate clearly** - Explain your thought process
