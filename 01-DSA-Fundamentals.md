# DSA Fundamentals - Complete Guide for FAANG Interviews

## Table of Contents
1. [Big O Notation](#big-o-notation)
2. [Linear Data Structures](#linear-data-structures)
   - Arrays
   - Linked Lists
   - Stacks
   - Queues
3. [Non-Linear Data Structures](#non-linear-data-structures)
   - Trees
   - Graphs
   - Heaps
   - Hash Tables
4. [Sorting Algorithms](#sorting-algorithms)
5. [Searching Algorithms](#searching-algorithms)

---

## Big O Notation

Big O describes the **worst-case** time/space complexity as input grows.

| Notation | Name | Example |
|----------|------|---------|
| O(1) | Constant | Array access by index |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Linear search |
| O(n log n) | Linearithmic | Merge sort |
| O(n²) | Quadratic | Bubble sort |
| O(2ⁿ) | Exponential | Recursive Fibonacci |
| O(n!) | Factorial | Permutations |

**Growth Rate (slowest to fastest):** O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) < O(n!)

---

## Linear Data Structures

### 1. Arrays

An array is a contiguous block of memory storing elements of the same type.

#### Operations & Time Complexities

| Operation | Time Complexity | Description |
|-----------|----------------|-------------|
| Access by index | O(1) | Direct memory access |
| Insert at end | O(1) amortized | May need resize |
| Insert at beginning | O(n) | Shift all elements right |
| Insert at middle | O(n) | Shift elements after index |
| Delete at end | O(1) | Simple removal |
| Delete at beginning | O(n) | Shift all elements left |
| Delete at middle | O(n) | Shift elements after index |
| Search (unsorted) | O(n) | Linear scan |
| Search (sorted) | O(log n) | Binary search |

#### Implementation Examples

**Java:**
```java
public class ArrayOperations {
    
    // Insert at beginning - O(n)
    public static int[] insertAtBeginning(int[] arr, int element) {
        int[] newArr = new int[arr.length + 1];
        newArr[0] = element;
        for (int i = 0; i < arr.length; i++) {
            newArr[i + 1] = arr[i];
        }
        return newArr;
    }
    
    // Insert at end - O(1) for ArrayList, O(n) for fixed array
    public static int[] insertAtEnd(int[] arr, int element) {
        int[] newArr = new int[arr.length + 1];
        for (int i = 0; i < arr.length; i++) {
            newArr[i] = arr[i];
        }
        newArr[arr.length] = element;
        return newArr;
    }
    
    // Insert at middle (at index) - O(n)
    public static int[] insertAtIndex(int[] arr, int index, int element) {
        int[] newArr = new int[arr.length + 1];
        for (int i = 0; i < index; i++) {
            newArr[i] = arr[i];
        }
        newArr[index] = element;
        for (int i = index; i < arr.length; i++) {
            newArr[i + 1] = arr[i];
        }
        return newArr;
    }
    
    // Delete at index - O(n)
    public static int[] deleteAtIndex(int[] arr, int index) {
        int[] newArr = new int[arr.length - 1];
        for (int i = 0, j = 0; i < arr.length; i++) {
            if (i != index) {
                newArr[j++] = arr[i];
            }
        }
        return newArr;
    }
    
    // Linear Search - O(n)
    public static int linearSearch(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) return i;
        }
        return -1;
    }
    
    // Binary Search (sorted array) - O(log n)
    public static int binarySearch(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }
}
```

**Python:**
```python
class ArrayOperations:
    
    # Insert at beginning - O(n)
    @staticmethod
    def insert_at_beginning(arr, element):
        return [element] + arr
    
    # Insert at end - O(1) amortized
    @staticmethod
    def insert_at_end(arr, element):
        arr.append(element)
        return arr
    
    # Insert at middle (at index) - O(n)
    @staticmethod
    def insert_at_index(arr, index, element):
        arr.insert(index, element)
        return arr
    
    # Delete at index - O(n)
    @staticmethod
    def delete_at_index(arr, index):
        arr.pop(index)
        return arr
    
    # Linear Search - O(n)
    @staticmethod
    def linear_search(arr, target):
        for i, val in enumerate(arr):
            if val == target:
                return i
        return -1
    
    # Binary Search (sorted array) - O(log n)
    @staticmethod
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

---

### 2. Linked Lists

A linked list is a linear data structure where elements are stored in nodes, each pointing to the next.

#### Types of Linked Lists
- **Singly Linked List**: Each node points to next only
- **Doubly Linked List**: Each node points to both next and previous
- **Circular Linked List**: Last node points back to first

#### Operations & Time Complexities

| Operation | Singly LL | Doubly LL |
|-----------|-----------|-----------|
| Access by index | O(n) | O(n) |
| Insert at head | O(1) | O(1) |
| Insert at tail | O(n) or O(1)* | O(1) |
| Insert at middle | O(n) | O(n) |
| Delete at head | O(1) | O(1) |
| Delete at tail | O(n) | O(1) |
| Delete at middle | O(n) | O(n) |
| Search | O(n) | O(n) |

*O(1) if tail pointer is maintained

#### Implementation Examples

**Java - Singly Linked List:**
```java
class ListNode {
    int val;
    ListNode next;
    
    ListNode(int val) {
        this.val = val;
        this.next = null;
    }
}

class SinglyLinkedList {
    private ListNode head;
    private ListNode tail;
    private int size;
    
    public SinglyLinkedList() {
        head = null;
        tail = null;
        size = 0;
    }
    
    // Insert at head - O(1)
    public void insertAtHead(int val) {
        ListNode newNode = new ListNode(val);
        if (head == null) {
            head = tail = newNode;
        } else {
            newNode.next = head;
            head = newNode;
        }
        size++;
    }
    
    // Insert at tail - O(1) with tail pointer
    public void insertAtTail(int val) {
        ListNode newNode = new ListNode(val);
        if (tail == null) {
            head = tail = newNode;
        } else {
            tail.next = newNode;
            tail = newNode;
        }
        size++;
    }
    
    // Insert at index - O(n)
    public void insertAtIndex(int index, int val) {
        if (index < 0 || index > size) return;
        if (index == 0) {
            insertAtHead(val);
            return;
        }
        if (index == size) {
            insertAtTail(val);
            return;
        }
        
        ListNode newNode = new ListNode(val);
        ListNode current = head;
        for (int i = 0; i < index - 1; i++) {
            current = current.next;
        }
        newNode.next = current.next;
        current.next = newNode;
        size++;
    }
    
    // Delete at head - O(1)
    public void deleteAtHead() {
        if (head == null) return;
        if (head == tail) {
            head = tail = null;
        } else {
            head = head.next;
        }
        size--;
    }
    
    // Delete at tail - O(n) for singly linked list
    public void deleteAtTail() {
        if (head == null) return;
        if (head == tail) {
            head = tail = null;
            size--;
            return;
        }
        
        ListNode current = head;
        while (current.next != tail) {
            current = current.next;
        }
        current.next = null;
        tail = current;
        size--;
    }
    
    // Delete at index - O(n)
    public void deleteAtIndex(int index) {
        if (index < 0 || index >= size) return;
        if (index == 0) {
            deleteAtHead();
            return;
        }
        
        ListNode current = head;
        for (int i = 0; i < index - 1; i++) {
            current = current.next;
        }
        
        if (current.next == tail) {
            tail = current;
        }
        current.next = current.next.next;
        size--;
    }
    
    // Search - O(n)
    public int search(int val) {
        ListNode current = head;
        int index = 0;
        while (current != null) {
            if (current.val == val) return index;
            current = current.next;
            index++;
        }
        return -1;
    }
    
    // Reverse - O(n)
    public void reverse() {
        ListNode prev = null;
        ListNode current = head;
        tail = head;
        
        while (current != null) {
            ListNode next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        head = prev;
    }
}
```

**Python - Singly Linked List:**
```python
class ListNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    # Insert at head - O(1)
    def insert_at_head(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node
        self.size += 1
    
    # Insert at tail - O(1) with tail pointer
    def insert_at_tail(self, val):
        new_node = ListNode(val)
        if not self.tail:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
    
    # Insert at index - O(n)
    def insert_at_index(self, index, val):
        if index < 0 or index > self.size:
            return
        if index == 0:
            self.insert_at_head(val)
            return
        if index == self.size:
            self.insert_at_tail(val)
            return
        
        new_node = ListNode(val)
        current = self.head
        for _ in range(index - 1):
            current = current.next
        new_node.next = current.next
        current.next = new_node
        self.size += 1
    
    # Delete at head - O(1)
    def delete_at_head(self):
        if not self.head:
            return
        if self.head == self.tail:
            self.head = self.tail = None
        else:
            self.head = self.head.next
        self.size -= 1
    
    # Delete at tail - O(n)
    def delete_at_tail(self):
        if not self.head:
            return
        if self.head == self.tail:
            self.head = self.tail = None
            self.size -= 1
            return
        
        current = self.head
        while current.next != self.tail:
            current = current.next
        current.next = None
        self.tail = current
        self.size -= 1
    
    # Delete at index - O(n)
    def delete_at_index(self, index):
        if index < 0 or index >= self.size:
            return
        if index == 0:
            self.delete_at_head()
            return
        
        current = self.head
        for _ in range(index - 1):
            current = current.next
        
        if current.next == self.tail:
            self.tail = current
        current.next = current.next.next
        self.size -= 1
    
    # Search - O(n)
    def search(self, val):
        current = self.head
        index = 0
        while current:
            if current.val == val:
                return index
            current = current.next
            index += 1
        return -1
    
    # Reverse - O(n)
    def reverse(self):
        prev = None
        current = self.head
        self.tail = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev
```

---

### 3. Stacks

A stack is a LIFO (Last In, First Out) data structure.

#### Operations & Time Complexities

| Operation | Time Complexity |
|-----------|----------------|
| Push | O(1) |
| Pop | O(1) |
| Peek/Top | O(1) |
| isEmpty | O(1) |
| Size | O(1) |
| Search | O(n) |

#### Implementation Examples

**Java:**
```java
import java.util.Stack;
import java.util.EmptyStackException;

// Using built-in Stack
class StackDemo {
    public static void main(String[] args) {
        Stack<Integer> stack = new Stack<>();
        
        stack.push(10);    // Push - O(1)
        stack.push(20);
        stack.push(30);
        
        int top = stack.peek();  // Peek - O(1), returns 30
        int popped = stack.pop(); // Pop - O(1), returns 30
        boolean empty = stack.isEmpty(); // isEmpty - O(1)
        int size = stack.size(); // Size - O(1)
        int index = stack.search(10); // Search - O(n)
    }
}

// Custom implementation using array
class ArrayStack {
    private int[] arr;
    private int top;
    private int capacity;
    
    public ArrayStack(int capacity) {
        this.capacity = capacity;
        arr = new int[capacity];
        top = -1;
    }
    
    // Push - O(1)
    public void push(int val) {
        if (top == capacity - 1) {
            throw new StackOverflowError("Stack is full");
        }
        arr[++top] = val;
    }
    
    // Pop - O(1)
    public int pop() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return arr[top--];
    }
    
    // Peek - O(1)
    public int peek() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return arr[top];
    }
    
    // isEmpty - O(1)
    public boolean isEmpty() {
        return top == -1;
    }
    
    // Size - O(1)
    public int size() {
        return top + 1;
    }
}
```

**Python:**
```python
# Using list as stack
stack = []

stack.append(10)    # Push - O(1) amortized
stack.append(20)
stack.append(30)

top = stack[-1]     # Peek - O(1), returns 30
popped = stack.pop() # Pop - O(1), returns 30
is_empty = len(stack) == 0  # isEmpty - O(1)
size = len(stack)   # Size - O(1)

# Using collections.deque (more efficient)
from collections import deque

class Stack:
    def __init__(self):
        self.stack = deque()
    
    # Push - O(1)
    def push(self, val):
        self.stack.append(val)
    
    # Pop - O(1)
    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.stack.pop()
    
    # Peek - O(1)
    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.stack[-1]
    
    # isEmpty - O(1)
    def is_empty(self):
        return len(self.stack) == 0
    
    # Size - O(1)
    def size(self):
        return len(self.stack)
```

---

### 4. Queues

A queue is a FIFO (First In, First Out) data structure.

#### Types of Queues
- **Simple Queue**: Basic FIFO
- **Circular Queue**: Last position connects to first
- **Priority Queue**: Elements dequeued by priority
- **Deque (Double-ended)**: Insert/delete from both ends

#### Operations & Time Complexities

| Operation | Queue | Deque | Priority Queue |
|-----------|-------|-------|----------------|
| Enqueue/Add | O(1) | O(1) | O(log n) |
| Dequeue/Remove | O(1) | O(1) | O(log n) |
| Peek/Front | O(1) | O(1) | O(1) |
| isEmpty | O(1) | O(1) | O(1) |

#### Implementation Examples

**Java:**
```java
import java.util.*;

class QueueDemo {
    public static void main(String[] args) {
        // Simple Queue using LinkedList
        Queue<Integer> queue = new LinkedList<>();
        
        queue.offer(10);    // Enqueue - O(1)
        queue.offer(20);
        queue.offer(30);
        
        int front = queue.peek();  // Peek - O(1), returns 10
        int removed = queue.poll(); // Dequeue - O(1), returns 10
        boolean empty = queue.isEmpty(); // isEmpty - O(1)
        
        // Deque (Double-ended queue)
        Deque<Integer> deque = new ArrayDeque<>();
        
        deque.offerFirst(10); // Add at front - O(1)
        deque.offerLast(20);  // Add at back - O(1)
        deque.pollFirst();    // Remove from front - O(1)
        deque.pollLast();     // Remove from back - O(1)
        
        // Priority Queue (Min-Heap by default)
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        
        minHeap.offer(30);   // Add - O(log n)
        minHeap.offer(10);
        minHeap.offer(20);
        
        int min = minHeap.peek();  // Peek - O(1), returns 10
        int removed2 = minHeap.poll(); // Remove - O(log n), returns 10
        
        // Max-Heap
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
    }
}

// Custom Queue using array (Circular Queue)
class CircularQueue {
    private int[] arr;
    private int front, rear, size, capacity;
    
    public CircularQueue(int capacity) {
        this.capacity = capacity;
        arr = new int[capacity];
        front = 0;
        rear = -1;
        size = 0;
    }
    
    // Enqueue - O(1)
    public void enqueue(int val) {
        if (size == capacity) {
            throw new IllegalStateException("Queue is full");
        }
        rear = (rear + 1) % capacity;
        arr[rear] = val;
        size++;
    }
    
    // Dequeue - O(1)
    public int dequeue() {
        if (isEmpty()) {
            throw new IllegalStateException("Queue is empty");
        }
        int val = arr[front];
        front = (front + 1) % capacity;
        size--;
        return val;
    }
    
    // Peek - O(1)
    public int peek() {
        if (isEmpty()) {
            throw new IllegalStateException("Queue is empty");
        }
        return arr[front];
    }
    
    // isEmpty - O(1)
    public boolean isEmpty() {
        return size == 0;
    }
}
```

**Python:**
```python
from collections import deque
import heapq

# Simple Queue using deque
queue = deque()

queue.append(10)    # Enqueue - O(1)
queue.append(20)
queue.append(30)

front = queue[0]    # Peek - O(1), returns 10
removed = queue.popleft()  # Dequeue - O(1), returns 10
is_empty = len(queue) == 0  # isEmpty - O(1)

# Deque (Double-ended queue)
dq = deque()

dq.appendleft(10)   # Add at front - O(1)
dq.append(20)       # Add at back - O(1)
dq.popleft()        # Remove from front - O(1)
dq.pop()            # Remove from back - O(1)

# Priority Queue using heapq (Min-Heap)
min_heap = []

heapq.heappush(min_heap, 30)  # Add - O(log n)
heapq.heappush(min_heap, 10)
heapq.heappush(min_heap, 20)

min_val = min_heap[0]          # Peek - O(1), returns 10
removed = heapq.heappop(min_heap)  # Remove - O(log n), returns 10

# Max-Heap (negate values)
max_heap = []
heapq.heappush(max_heap, -30)
heapq.heappush(max_heap, -10)
max_val = -max_heap[0]  # Returns 30

# Custom Queue class
class Queue:
    def __init__(self):
        self.queue = deque()
    
    def enqueue(self, val):
        self.queue.append(val)
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.queue.popleft()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.queue[0]
    
    def is_empty(self):
        return len(self.queue) == 0
    
    def size(self):
        return len(self.queue)
```

---

## Non-Linear Data Structures

### 1. Hash Tables (Hash Maps)

A hash table stores key-value pairs with near-constant time operations using a hash function.

#### Operations & Time Complexities

| Operation | Average | Worst (with collisions) |
|-----------|---------|------------------------|
| Insert | O(1) | O(n) |
| Delete | O(1) | O(n) |
| Search | O(1) | O(n) |
| Access | O(1) | O(n) |

#### Implementation Examples

**Java:**
```java
import java.util.*;

class HashMapDemo {
    public static void main(String[] args) {
        // Using HashMap
        HashMap<String, Integer> map = new HashMap<>();
        
        // Insert - O(1) average
        map.put("apple", 5);
        map.put("banana", 3);
        map.put("orange", 7);
        
        // Access - O(1) average
        int value = map.get("apple");  // Returns 5
        int defaultVal = map.getOrDefault("grape", 0);  // Returns 0
        
        // Check existence - O(1) average
        boolean hasKey = map.containsKey("apple");  // true
        boolean hasValue = map.containsValue(5);    // true (O(n))
        
        // Delete - O(1) average
        map.remove("banana");
        
        // Iterate - O(n)
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
        
        // LinkedHashMap maintains insertion order
        LinkedHashMap<String, Integer> orderedMap = new LinkedHashMap<>();
        
        // TreeMap maintains sorted order - O(log n) operations
        TreeMap<String, Integer> sortedMap = new TreeMap<>();
    }
}

// Simple HashSet usage
class HashSetDemo {
    public static void main(String[] args) {
        HashSet<Integer> set = new HashSet<>();
        
        set.add(10);      // Add - O(1) average
        set.add(20);
        set.remove(10);   // Remove - O(1) average
        boolean has = set.contains(20);  // Contains - O(1) average
    }
}
```

**Python:**
```python
# Using dict (built-in hash map)
hash_map = {}

# Insert - O(1) average
hash_map["apple"] = 5
hash_map["banana"] = 3
hash_map["orange"] = 7

# Access - O(1) average
value = hash_map["apple"]  # Returns 5
default_val = hash_map.get("grape", 0)  # Returns 0 (default)

# Check existence - O(1) average
has_key = "apple" in hash_map  # True
has_value = 5 in hash_map.values()  # True (O(n))

# Delete - O(1) average
del hash_map["banana"]
# or
hash_map.pop("orange", None)  # Returns value or None

# Iterate - O(n)
for key, value in hash_map.items():
    print(f"{key}: {value}")

# Using set
hash_set = set()

hash_set.add(10)      # Add - O(1) average
hash_set.add(20)
hash_set.remove(10)   # Remove - O(1) average (raises KeyError if not found)
hash_set.discard(30)  # Remove - O(1) (no error if not found)
has = 20 in hash_set  # Contains - O(1) average

# Counter for frequency counting
from collections import Counter

nums = [1, 2, 2, 3, 3, 3]
freq = Counter(nums)  # {3: 3, 2: 2, 1: 1}

# defaultdict for automatic default values
from collections import defaultdict

graph = defaultdict(list)
graph[1].append(2)  # No KeyError even if key doesn't exist
```

---

### 2. Trees

A tree is a hierarchical data structure with nodes connected by edges.

#### Types of Trees
- **Binary Tree**: Each node has at most 2 children
- **Binary Search Tree (BST)**: Left child < parent < right child
- **AVL Tree**: Self-balancing BST
- **Red-Black Tree**: Self-balancing BST
- **Trie**: Prefix tree for strings

#### Binary Tree Operations & Time Complexities

| Operation | BST Average | BST Worst | Balanced BST |
|-----------|-------------|-----------|--------------|
| Search | O(log n) | O(n) | O(log n) |
| Insert | O(log n) | O(n) | O(log n) |
| Delete | O(log n) | O(n) | O(log n) |
| Traversal | O(n) | O(n) | O(n) |

#### Implementation Examples

**Java:**
```java
class TreeNode {
    int val;
    TreeNode left, right;
    
    TreeNode(int val) {
        this.val = val;
        left = right = null;
    }
}

class BinarySearchTree {
    private TreeNode root;
    
    // Insert - O(log n) average, O(n) worst
    public void insert(int val) {
        root = insertRec(root, val);
    }
    
    private TreeNode insertRec(TreeNode node, int val) {
        if (node == null) return new TreeNode(val);
        
        if (val < node.val) {
            node.left = insertRec(node.left, val);
        } else if (val > node.val) {
            node.right = insertRec(node.right, val);
        }
        return node;
    }
    
    // Search - O(log n) average, O(n) worst
    public boolean search(int val) {
        return searchRec(root, val);
    }
    
    private boolean searchRec(TreeNode node, int val) {
        if (node == null) return false;
        if (node.val == val) return true;
        
        if (val < node.val) {
            return searchRec(node.left, val);
        }
        return searchRec(node.right, val);
    }
    
    // Delete - O(log n) average, O(n) worst
    public void delete(int val) {
        root = deleteRec(root, val);
    }
    
    private TreeNode deleteRec(TreeNode node, int val) {
        if (node == null) return null;
        
        if (val < node.val) {
            node.left = deleteRec(node.left, val);
        } else if (val > node.val) {
            node.right = deleteRec(node.right, val);
        } else {
            // Node with one or no child
            if (node.left == null) return node.right;
            if (node.right == null) return node.left;
            
            // Node with two children: Get inorder successor
            node.val = minValue(node.right);
            node.right = deleteRec(node.right, node.val);
        }
        return node;
    }
    
    private int minValue(TreeNode node) {
        int min = node.val;
        while (node.left != null) {
            min = node.left.val;
            node = node.left;
        }
        return min;
    }
    
    // Tree Traversals - All O(n)
    
    // Inorder: Left -> Root -> Right (gives sorted order for BST)
    public void inorder(TreeNode node) {
        if (node != null) {
            inorder(node.left);
            System.out.print(node.val + " ");
            inorder(node.right);
        }
    }
    
    // Preorder: Root -> Left -> Right
    public void preorder(TreeNode node) {
        if (node != null) {
            System.out.print(node.val + " ");
            preorder(node.left);
            preorder(node.right);
        }
    }
    
    // Postorder: Left -> Right -> Root
    public void postorder(TreeNode node) {
        if (node != null) {
            postorder(node.left);
            postorder(node.right);
            System.out.print(node.val + " ");
        }
    }
    
    // Level Order (BFS) - O(n)
    public void levelOrder() {
        if (root == null) return;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            System.out.print(node.val + " ");
            
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
    }
    
    // Height of tree - O(n)
    public int height(TreeNode node) {
        if (node == null) return -1;  // or 0 depending on definition
        return 1 + Math.max(height(node.left), height(node.right));
    }
}
```

**Python:**
```python
from collections import deque

class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    # Insert - O(log n) average, O(n) worst
    def insert(self, val):
        self.root = self._insert_rec(self.root, val)
    
    def _insert_rec(self, node, val):
        if not node:
            return TreeNode(val)
        
        if val < node.val:
            node.left = self._insert_rec(node.left, val)
        elif val > node.val:
            node.right = self._insert_rec(node.right, val)
        return node
    
    # Search - O(log n) average, O(n) worst
    def search(self, val):
        return self._search_rec(self.root, val)
    
    def _search_rec(self, node, val):
        if not node:
            return False
        if node.val == val:
            return True
        
        if val < node.val:
            return self._search_rec(node.left, val)
        return self._search_rec(node.right, val)
    
    # Delete - O(log n) average, O(n) worst
    def delete(self, val):
        self.root = self._delete_rec(self.root, val)
    
    def _delete_rec(self, node, val):
        if not node:
            return None
        
        if val < node.val:
            node.left = self._delete_rec(node.left, val)
        elif val > node.val:
            node.right = self._delete_rec(node.right, val)
        else:
            # Node with one or no child
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            
            # Node with two children
            node.val = self._min_value(node.right)
            node.right = self._delete_rec(node.right, node.val)
        return node
    
    def _min_value(self, node):
        current = node
        while current.left:
            current = current.left
        return current.val
    
    # Tree Traversals - All O(n)
    
    # Inorder: Left -> Root -> Right
    def inorder(self, node, result=None):
        if result is None:
            result = []
        if node:
            self.inorder(node.left, result)
            result.append(node.val)
            self.inorder(node.right, result)
        return result
    
    # Preorder: Root -> Left -> Right
    def preorder(self, node, result=None):
        if result is None:
            result = []
        if node:
            result.append(node.val)
            self.preorder(node.left, result)
            self.preorder(node.right, result)
        return result
    
    # Postorder: Left -> Right -> Root
    def postorder(self, node, result=None):
        if result is None:
            result = []
        if node:
            self.postorder(node.left, result)
            self.postorder(node.right, result)
            result.append(node.val)
        return result
    
    # Level Order (BFS) - O(n)
    def level_order(self):
        if not self.root:
            return []
        
        result = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return result
    
    # Height of tree - O(n)
    def height(self, node):
        if not node:
            return -1  # or 0 depending on definition
        return 1 + max(self.height(node.left), self.height(node.right))
```

---

### 3. Heaps

A heap is a complete binary tree that satisfies the heap property (min-heap: parent ≤ children, max-heap: parent ≥ children).

#### Operations & Time Complexities

| Operation | Time Complexity |
|-----------|----------------|
| Insert (push) | O(log n) |
| Extract min/max (pop) | O(log n) |
| Peek min/max | O(1) |
| Build heap | O(n) |
| Heapify | O(log n) |

**Java:**
```java
import java.util.*;

class HeapDemo {
    public static void main(String[] args) {
        // Min Heap
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        
        // Max Heap
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
        
        // Custom comparator
        PriorityQueue<int[]> customHeap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
    }
}
```

**Python:**
```python
import heapq

# Min Heap
min_heap = []
heapq.heappush(min_heap, 5)
heapq.heappush(min_heap, 3)
heapq.heappush(min_heap, 7)

min_val = heapq.heappop(min_heap)  # Returns 3

# Build heap from list - O(n)
arr = [5, 3, 7, 1, 9]
heapq.heapify(arr)  # Converts arr to min heap in-place

# Max Heap (negate values)
max_heap = []
heapq.heappush(max_heap, -5)
max_val = -heapq.heappop(max_heap)  # Returns 5

# Get n smallest/largest - O(n log k)
arr = [5, 3, 7, 1, 9, 2, 8]
smallest_3 = heapq.nsmallest(3, arr)  # [1, 2, 3]
largest_3 = heapq.nlargest(3, arr)    # [9, 8, 7]
```

---

### 4. Graphs

A graph consists of vertices (nodes) connected by edges.

#### Types of Graphs
- **Directed vs Undirected**
- **Weighted vs Unweighted**
- **Cyclic vs Acyclic**
- **Connected vs Disconnected**

#### Graph Representations

**1. Adjacency Matrix** - O(V²) space
**2. Adjacency List** - O(V + E) space (preferred for sparse graphs)

#### Operations & Time Complexities

| Operation | Adjacency Matrix | Adjacency List |
|-----------|-----------------|----------------|
| Add Edge | O(1) | O(1) |
| Remove Edge | O(1) | O(E) |
| Check Edge | O(1) | O(V) |
| Find Neighbors | O(V) | O(degree) |
| BFS/DFS | O(V²) | O(V + E) |
| Space | O(V²) | O(V + E) |

#### Implementation Examples

**Java:**
```java
import java.util.*;

class Graph {
    private Map<Integer, List<Integer>> adjList;
    
    public Graph() {
        adjList = new HashMap<>();
    }
    
    // Add vertex - O(1)
    public void addVertex(int v) {
        adjList.putIfAbsent(v, new ArrayList<>());
    }
    
    // Add edge (undirected) - O(1)
    public void addEdge(int u, int v) {
        adjList.putIfAbsent(u, new ArrayList<>());
        adjList.putIfAbsent(v, new ArrayList<>());
        adjList.get(u).add(v);
        adjList.get(v).add(u);
    }
    
    // BFS - O(V + E)
    public List<Integer> bfs(int start) {
        List<Integer> result = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        
        visited.add(start);
        queue.offer(start);
        
        while (!queue.isEmpty()) {
            int node = queue.poll();
            result.add(node);
            
            for (int neighbor : adjList.getOrDefault(node, new ArrayList<>())) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.offer(neighbor);
                }
            }
        }
        return result;
    }
    
    // DFS - O(V + E)
    public List<Integer> dfs(int start) {
        List<Integer> result = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();
        dfsHelper(start, visited, result);
        return result;
    }
    
    private void dfsHelper(int node, Set<Integer> visited, List<Integer> result) {
        visited.add(node);
        result.add(node);
        
        for (int neighbor : adjList.getOrDefault(node, new ArrayList<>())) {
            if (!visited.contains(neighbor)) {
                dfsHelper(neighbor, visited, result);
            }
        }
    }
    
    // Iterative DFS using Stack
    public List<Integer> dfsIterative(int start) {
        List<Integer> result = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();
        Stack<Integer> stack = new Stack<>();
        
        stack.push(start);
        
        while (!stack.isEmpty()) {
            int node = stack.pop();
            
            if (!visited.contains(node)) {
                visited.add(node);
                result.add(node);
                
                for (int neighbor : adjList.getOrDefault(node, new ArrayList<>())) {
                    if (!visited.contains(neighbor)) {
                        stack.push(neighbor);
                    }
                }
            }
        }
        return result;
    }
    
    // Detect cycle in undirected graph - O(V + E)
    public boolean hasCycle() {
        Set<Integer> visited = new HashSet<>();
        
        for (int node : adjList.keySet()) {
            if (!visited.contains(node)) {
                if (hasCycleDFS(node, -1, visited)) {
                    return true;
                }
            }
        }
        return false;
    }
    
    private boolean hasCycleDFS(int node, int parent, Set<Integer> visited) {
        visited.add(node);
        
        for (int neighbor : adjList.getOrDefault(node, new ArrayList<>())) {
            if (!visited.contains(neighbor)) {
                if (hasCycleDFS(neighbor, node, visited)) {
                    return true;
                }
            } else if (neighbor != parent) {
                return true;
            }
        }
        return false;
    }
}
```

**Python:**
```python
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.adj_list = defaultdict(list)
    
    # Add edge (undirected) - O(1)
    def add_edge(self, u, v):
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)
    
    # BFS - O(V + E)
    def bfs(self, start):
        result = []
        visited = set([start])
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in self.adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    # DFS Recursive - O(V + E)
    def dfs(self, start):
        result = []
        visited = set()
        self._dfs_helper(start, visited, result)
        return result
    
    def _dfs_helper(self, node, visited, result):
        visited.add(node)
        result.append(node)
        
        for neighbor in self.adj_list[node]:
            if neighbor not in visited:
                self._dfs_helper(neighbor, visited, result)
    
    # DFS Iterative using Stack - O(V + E)
    def dfs_iterative(self, start):
        result = []
        visited = set()
        stack = [start]
        
        while stack:
            node = stack.pop()
            
            if node not in visited:
                visited.add(node)
                result.append(node)
                
                for neighbor in self.adj_list[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return result
    
    # Detect cycle in undirected graph - O(V + E)
    def has_cycle(self):
        visited = set()
        
        for node in self.adj_list:
            if node not in visited:
                if self._has_cycle_dfs(node, -1, visited):
                    return True
        return False
    
    def _has_cycle_dfs(self, node, parent, visited):
        visited.add(node)
        
        for neighbor in self.adj_list[node]:
            if neighbor not in visited:
                if self._has_cycle_dfs(neighbor, node, visited):
                    return True
            elif neighbor != parent:
                return True
        return False
```

---

### 5. Trie (Prefix Tree)

A trie is a tree-like data structure for storing strings, enabling fast prefix-based operations.

#### Operations & Time Complexities

| Operation | Time Complexity |
|-----------|----------------|
| Insert | O(m) where m = word length |
| Search | O(m) |
| Prefix Search | O(m) |
| Delete | O(m) |

**Java:**
```java
class TrieNode {
    TrieNode[] children;
    boolean isEndOfWord;
    
    TrieNode() {
        children = new TrieNode[26];
        isEndOfWord = false;
    }
}

class Trie {
    private TrieNode root;
    
    public Trie() {
        root = new TrieNode();
    }
    
    // Insert - O(m)
    public void insert(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) {
                node.children[index] = new TrieNode();
            }
            node = node.children[index];
        }
        node.isEndOfWord = true;
    }
    
    // Search - O(m)
    public boolean search(String word) {
        TrieNode node = searchPrefix(word);
        return node != null && node.isEndOfWord;
    }
    
    // Starts With (Prefix Search) - O(m)
    public boolean startsWith(String prefix) {
        return searchPrefix(prefix) != null;
    }
    
    private TrieNode searchPrefix(String prefix) {
        TrieNode node = root;
        for (char c : prefix.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) {
                return null;
            }
            node = node.children[index];
        }
        return node;
    }
}
```

**Python:**
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    # Insert - O(m)
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    # Search - O(m)
    def search(self, word):
        node = self._search_prefix(word)
        return node is not None and node.is_end_of_word
    
    # Starts With (Prefix Search) - O(m)
    def starts_with(self, prefix):
        return self._search_prefix(prefix) is not None
    
    def _search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

---

## Sorting Algorithms

### Comparison of Sorting Algorithms

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| Counting Sort | O(n + k) | O(n + k) | O(n + k) | O(k) | Yes |
| Radix Sort | O(nk) | O(nk) | O(nk) | O(n + k) | Yes |

### Implementation Examples

**Java:**
```java
import java.util.*;

class SortingAlgorithms {
    
    // Bubble Sort - O(n²)
    public static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            boolean swapped = false;
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                    swapped = true;
                }
            }
            if (!swapped) break;  // Optimization: already sorted
        }
    }
    
    // Selection Sort - O(n²)
    public static void selectionSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIdx]) {
                    minIdx = j;
                }
            }
            int temp = arr[minIdx];
            arr[minIdx] = arr[i];
            arr[i] = temp;
        }
    }
    
    // Insertion Sort - O(n²)
    public static void insertionSort(int[] arr) {
        int n = arr.length;
        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
    
    // Merge Sort - O(n log n)
    public static void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            merge(arr, left, mid, right);
        }
    }
    
    private static void merge(int[] arr, int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;
        
        int[] L = new int[n1];
        int[] R = new int[n2];
        
        for (int i = 0; i < n1; i++) L[i] = arr[left + i];
        for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];
        
        int i = 0, j = 0, k = left;
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) {
                arr[k++] = L[i++];
            } else {
                arr[k++] = R[j++];
            }
        }
        while (i < n1) arr[k++] = L[i++];
        while (j < n2) arr[k++] = R[j++];
    }
    
    // Quick Sort - O(n log n) average, O(n²) worst
    public static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }
    
    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        
        return i + 1;
    }
    
    // Heap Sort - O(n log n)
    public static void heapSort(int[] arr) {
        int n = arr.length;
        
        // Build max heap
        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(arr, n, i);
        }
        
        // Extract elements from heap
        for (int i = n - 1; i > 0; i--) {
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;
            heapify(arr, i, 0);
        }
    }
    
    private static void heapify(int[] arr, int n, int i) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        if (left < n && arr[left] > arr[largest]) largest = left;
        if (right < n && arr[right] > arr[largest]) largest = right;
        
        if (largest != i) {
            int swap = arr[i];
            arr[i] = arr[largest];
            arr[largest] = swap;
            heapify(arr, n, largest);
        }
    }
}
```

**Python:**
```python
# Bubble Sort - O(n²)
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

# Selection Sort - O(n²)
def selection_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# Insertion Sort - O(n²)
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# Merge Sort - O(n log n)
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Quick Sort - O(n log n) average
def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)
    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Heap Sort - O(n log n)
def heap_sort(arr):
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    
    return arr

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

# Python built-in sort - O(n log n) using Timsort
arr = [5, 2, 8, 1, 9]
arr.sort()  # In-place
sorted_arr = sorted(arr)  # Returns new sorted list
```

---

## Searching Algorithms

### Comparison

| Algorithm | Time Complexity | Space | Requirement |
|-----------|----------------|-------|-------------|
| Linear Search | O(n) | O(1) | None |
| Binary Search | O(log n) | O(1) | Sorted array |
| Jump Search | O(√n) | O(1) | Sorted array |
| Interpolation Search | O(log log n)* | O(1) | Sorted, uniform |
| Exponential Search | O(log n) | O(1) | Sorted array |

*Average case for uniformly distributed data

### Binary Search Variations

**Java:**
```java
class BinarySearchVariations {
    
    // Standard Binary Search
    public int binarySearch(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }
    
    // Find First Occurrence (Lower Bound)
    public int findFirst(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        int result = -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) {
                result = mid;
                right = mid - 1;  // Keep searching left
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return result;
    }
    
    // Find Last Occurrence (Upper Bound)
    public int findLast(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        int result = -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) {
                result = mid;
                left = mid + 1;  // Keep searching right
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return result;
    }
    
    // Search in Rotated Sorted Array
    public int searchRotated(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) return mid;
            
            // Left half is sorted
            if (arr[left] <= arr[mid]) {
                if (target >= arr[left] && target < arr[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            // Right half is sorted
            else {
                if (target > arr[mid] && target <= arr[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }
    
    // Find Peak Element
    public int findPeak(int[] arr) {
        int left = 0, right = arr.length - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] > arr[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
}
```

**Python:**
```python
import bisect

# Standard Binary Search
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

# Find First Occurrence (Lower Bound)
def find_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Keep searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

# Using bisect module
def find_first_bisect(arr, target):
    idx = bisect.bisect_left(arr, target)
    if idx < len(arr) and arr[idx] == target:
        return idx
    return -1

# Find Last Occurrence (Upper Bound)
def find_last(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Keep searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

# Search in Rotated Sorted Array
def search_rotated(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        
        # Left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

# Find Peak Element
def find_peak(arr):
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left
```

---

## Summary: When to Use What

| Data Structure | Best For |
|----------------|----------|
| Array | Random access, fixed size data |
| Linked List | Frequent insertions/deletions at ends |
| Stack | Undo operations, expression evaluation, DFS |
| Queue | BFS, scheduling, buffers |
| Hash Table | Fast lookups, counting, caching |
| BST | Ordered data with dynamic operations |
| Heap | Priority queues, finding min/max |
| Trie | String prefix operations, autocomplete |
| Graph | Networks, relationships, paths |

| Algorithm | Best For |
|-----------|----------|
| Merge Sort | Stable sort needed, linked lists |
| Quick Sort | Arrays, average case performance |
| Heap Sort | Limited memory, guaranteed O(n log n) |
| Counting Sort | Small range integers |
| Binary Search | Sorted array lookups |
| BFS | Shortest path (unweighted), level order |
| DFS | Path finding, cycle detection, topological sort |
