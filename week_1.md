# 🚀 Week 1: Your FAANG Journey Begins!
*Advanced Java Concepts + First Steps*
#### Visagan S

## 📅 Week 1 Overview
**Focus**: Java Collections Framework & HashMap Internals
**Goal**: Master Java fundamentals that are frequently asked in interviews
**Total Time**: 11 hours (5 weekday + 6 weekend)

---

## 🗓️ Daily Breakdown

### 📆 Day 1 - Monday (1 Hour)
**Topic: Java Collections Overview & ArrayList Deep Dive**

#### ⏰ Time Breakdown:
- **0-10 min**: Setup & Motivation
  - Create a dedicated folder: `FAANG_Prep/Week1/`
  - Create `Notes.md` file for notes
  - Create `Code/` folder for implementations
  - Open YouTube, bookmark channels

- **10-40 min**: Watch & Take Notes
  - 📹 Watch: [Java Collections Framework Overview](https://www.youtube.com/watch?v=GdAon80-0KA) (15 min)
  - 📹 Watch: [ArrayList Internal Working](https://www.youtube.com/watch?v=g2PDHB6kseU) (15 min)
  - Take notes on:
    - Collections hierarchy
    - When to use ArrayList vs LinkedList
    - Time complexities
    - Dynamic array resizing

- **40-60 min**: Code Implementation
  ```java
  // Implement: MyArrayList.java
  public class MyArrayList<T> {
      private Object[] data;
      private int size = 0;
      private static final int INITIAL_CAPACITY = 10;
      
      public MyArrayList() {
          data = new Object[INITIAL_CAPACITY];
      }
      
      public void add(T element) {
          // TODO: Implement add with resizing
          if (size == data.length) {
              resize();
          }
          data[size++] = element;
      }
      
      private void resize() {
          // TODO: Double the capacity
          Object[] newData = new Object[data.length * 2];
          System.arraycopy(data, 0, newData, 0, data.length);
          data = newData;
      }
      
      public T get(int index) {
          // TODO: Implement with bounds checking
          if (index < 0 || index >= size) {
              throw new IndexOutOfBoundsException();
          }
          return (T) data[index];
      }
      
      // Implement: remove(), size(), isEmpty()
  }
  ```

#### 📝 Day 1 Checklist:
- [ ] Watch both videos
- [ ] Take notes on Collections hierarchy
- [ ] Implement MyArrayList with at least 3 methods
- [ ] Test your implementation
- [ ] Write down 2 questions you have

#### 💡 Interview Tips for Day 1:
- ArrayList resizing happens at 50% capacity increase in Java (not doubling)
- Default initial capacity is 10
- Always mention amortized O(1) for add operation

---

### 📆 Day 2 - Tuesday (1 Hour)
**Topic: HashMap - The Most Important Data Structure**

#### ⏰ Time Breakdown:
- **0-5 min**: Quick Review
  - Review yesterday's ArrayList notes
  - Check if your implementation works

- **5-35 min**: HashMap Deep Dive
  - 📹 Watch: [HashMap Internal Working](https://www.youtube.com/watch?v=c3RVW3KGIIE) (30 min)
  - **Critical Notes to Take**:
    - How hashCode() works
    - Collision resolution (chaining vs open addressing)
    - Load factor (0.75) and rehashing
    - Bucket array structure
    - equals() vs hashCode() contract

- **35-60 min**: Coding Practice
  ```java
  // Practice Problems:
  
  // 1. Two Sum using HashMap (MUST KNOW!)
  public int[] twoSum(int[] nums, int target) {
      Map<Integer, Integer> map = new HashMap<>();
      for (int i = 0; i < nums.length; i++) {
          int complement = target - nums[i];
          if (map.containsKey(complement)) {
              return new int[] {map.get(complement), i};
          }
          map.put(nums[i], i);
      }
      return new int[] {-1, -1};
  }
  
  // 2. Find First Non-Repeating Character
  public char firstNonRepeating(String s) {
      Map<Character, Integer> count = new HashMap<>();
      // First pass: count frequencies
      for (char c : s.toCharArray()) {
          count.put(c, count.getOrDefault(c, 0) + 1);
      }
      // Second pass: find first with count 1
      for (char c : s.toCharArray()) {
          if (count.get(c) == 1) {
              return c;
          }
      }
      return '\0';
  }
  
  // 3. Group Anagrams
  public List<List<String>> groupAnagrams(String[] strs) {
      // TODO: Implement using HashMap
      // Hint: Use sorted string as key
  }
  ```

#### 📝 Day 2 Checklist:
- [ ] Understand HashMap internal working
- [ ] Solve Two Sum problem
- [ ] Solve at least one more HashMap problem
- [ ] Note down HashMap vs TreeMap vs LinkedHashMap differences

---

### 📆 Day 3 - Wednesday (1 Hour)
**Topic: HashMap Implementation & Collision Handling**

#### ⏰ Time Breakdown:
- **0-30 min**: Implement Basic HashMap
  ```java
  public class MyHashMap<K, V> {
      class Node<K, V> {
          K key;
          V value;
          Node<K, V> next;
          
          Node(K key, V value) {
              this.key = key;
              this.value = value;
          }
      }
      
      private Node<K, V>[] buckets;
      private int capacity = 16;
      private int size = 0;
      private final double LOAD_FACTOR = 0.75;
      
      public MyHashMap() {
          buckets = new Node[capacity];
      }
      
      private int getBucketIndex(K key) {
          int hashCode = key.hashCode();
          return Math.abs(hashCode) % capacity;
      }
      
      public void put(K key, V value) {
          int index = getBucketIndex(key);
          Node<K, V> head = buckets[index];
          
          // Check if key exists
          while (head != null) {
              if (head.key.equals(key)) {
                  head.value = value;
                  return;
              }
              head = head.next;
          }
          
          // Add new node at beginning
          Node<K, V> newNode = new Node<>(key, value);
          newNode.next = buckets[index];
          buckets[index] = newNode;
          size++;
          
          // Check load factor
          if (size > capacity * LOAD_FACTOR) {
              rehash();
          }
      }
      
      public V get(K key) {
          int index = getBucketIndex(key);
          Node<K, V> head = buckets[index];
          
          while (head != null) {
              if (head.key.equals(key)) {
                  return head.value;
              }
              head = head.next;
          }
          return null;
      }
      
      private void rehash() {
          // TODO: Implement rehashing
          System.out.println("Rehashing needed!");
      }
  }
  ```

- **30-45 min**: Test Your Implementation
  ```java
  public static void main(String[] args) {
      MyHashMap<String, Integer> map = new MyHashMap<>();
      
      // Test basic operations
      map.put("apple", 100);
      map.put("banana", 200);
      map.put("orange", 300);
      
      System.out.println(map.get("apple"));  // Should print 100
      System.out.println(map.get("grape"));  // Should print null
      
      // Test collision (if hash codes collide)
      map.put("FB", 500);  // "FB" and "Ea" have same hashCode!
      map.put("Ea", 600);
      
      System.out.println(map.get("FB"));
      System.out.println(map.get("Ea"));
  }
  ```

- **45-60 min**: LeetCode Easy Problem
  - Solve: [LeetCode #1: Two Sum](https://leetcode.com/problems/two-sum/)
  - Try to optimize from O(n²) to O(n) using HashMap

#### 📝 Day 3 Checklist:
- [ ] Implement basic HashMap with put() and get()
- [ ] Understand collision handling via chaining
- [ ] Test with collision cases
- [ ] Solve LeetCode #1

---

### 📆 Day 4 - Thursday (1 Hour)
**Topic: ConcurrentHashMap & Thread Safety**

#### ⏰ Time Breakdown:
- **0-30 min**: Understanding Thread Safety
  - 📹 Watch: [ConcurrentHashMap Internals](https://www.youtube.com/watch?v=FBjEn0UAUI4) (20 min)
  - 📹 Quick read: [HashMap vs ConcurrentHashMap](https://www.baeldung.com/java-concurrent-map) (10 min)
  - **Key Concepts**:
    - Segment-based locking (Java 7)
    - Node-based locking (Java 8+)
    - No null keys or values
    - Weakly consistent iterators

- **30-50 min**: Code Examples
  ```java
  // Thread-Safety Issues with HashMap
  public class HashMapThreadTest {
      public static void main(String[] args) throws InterruptedException {
          // DON'T DO THIS - HashMap is not thread-safe!
          Map<String, Integer> hashMap = new HashMap<>();
          
          // USE THIS - ConcurrentHashMap
          ConcurrentHashMap<String, Integer> concurrentMap = new ConcurrentHashMap<>();
          
          // Create multiple threads
          Thread t1 = new Thread(() -> {
              for (int i = 0; i < 1000; i++) {
                  concurrentMap.put("key" + i, i);
              }
          });
          
          Thread t2 = new Thread(() -> {
              for (int i = 1000; i < 2000; i++) {
                  concurrentMap.put("key" + i, i);
              }
          });
          
          t1.start();
          t2.start();
          t1.join();
          t2.join();
          
          System.out.println("Size: " + concurrentMap.size()); // Should be 2000
          
          // Atomic operations
          concurrentMap.putIfAbsent("newKey", 100);
          concurrentMap.compute("newKey", (k, v) -> v + 1);
          concurrentMap.merge("newKey", 10, Integer::sum);
      }
  }
  
  // Interview Question: Implement Thread-Safe Counter
  class ThreadSafeCounter {
      private ConcurrentHashMap<String, AtomicInteger> counters = new ConcurrentHashMap<>();
      
      public void increment(String key) {
          counters.computeIfAbsent(key, k -> new AtomicInteger(0))
                  .incrementAndGet();
      }
      
      public int getCount(String key) {
          AtomicInteger counter = counters.get(key);
          return counter == null ? 0 : counter.get();
      }
  }
  ```

- **50-60 min**: Practice Problem
  - Implement a simple frequency counter using ConcurrentHashMap
  - Make it thread-safe for multiple threads counting words

#### 📝 Day 4 Checklist:
- [ ] Understand HashMap vs ConcurrentHashMap
- [ ] Know when to use thread-safe collections
- [ ] Implement thread-safe counter
- [ ] Understand putIfAbsent, compute, merge operations

---

### 📆 Day 5 - Friday (1 Hour)
**Topic: Review & LeetCode Practice**

#### ⏰ Time Breakdown:
- **0-15 min**: Week Review
  - Review all notes from Mon-Thu
  - List 5 key concepts learned:
    1. ArrayList resizing mechanism
    2. HashMap collision handling
    3. Load factor and rehashing
    4. ConcurrentHashMap for thread safety
    5. Time complexities of operations

- **15-45 min**: Solve 2 LeetCode Problems
  ```java
  // Problem 1: Contains Duplicate (Easy)
  // https://leetcode.com/problems/contains-duplicate/
  public boolean containsDuplicate(int[] nums) {
      Set<Integer> set = new HashSet<>();
      for (int num : nums) {
          if (!set.add(num)) {  // add returns false if element exists
              return true;
          }
      }
      return false;
  }
  
  // Problem 2: Valid Anagram (Easy)
  // https://leetcode.com/problems/valid-anagram/
  public boolean isAnagram(String s, String t) {
      if (s.length() != t.length()) return false;
      
      Map<Character, Integer> count = new HashMap<>();
      
      // Count characters in s
      for (char c : s.toCharArray()) {
          count.put(c, count.getOrDefault(c, 0) + 1);
      }
      
      // Decrement for characters in t
      for (char c : t.toCharArray()) {
          int cnt = count.getOrDefault(c, 0);
          if (cnt == 0) return false;
          count.put(c, cnt - 1);
      }
      
      return true;
  }
  ```

- **45-60 min**: Prepare for Weekend
  - Set up environment for Saturday's deep dive
  - Download/bookmark resources for LinkedList
  - Review what you struggled with this week
  - Write down 3 questions to research on weekend

#### 📝 Day 5 Checklist:
- [ ] Complete week notes summary
- [ ] Solve both LeetCode problems
- [ ] Identify weak areas to focus on weekend
- [ ] Prepare Saturday's study materials

---

## 🎯 Weekend Deep Dive Sessions

### 📆 Saturday (2-3 Hours)
**Topic: Complete Data Structure Implementation + Problem Solving**

#### ⏰ Session 1 (1.5 Hours): LinkedList Mastery

**0-45 min: Watch & Learn**
- 📹 [LinkedList Complete Tutorial](https://www.youtube.com/watch?v=Hj_rA0dhr2I) (30 min)
- 📹 [Reverse a LinkedList](https://www.youtube.com/watch?v=iRtLEoL-r-g) (15 min)

**45-90 min: Implement Complete LinkedList**
```java
public class MyLinkedList<T> {
    class Node {
        T data;
        Node next;
        Node(T data) {
            this.data = data;
        }
    }
    
    private Node head;
    private int size;
    
    // Core Operations to Implement:
    public void addFirst(T data) {
        Node newNode = new Node(data);
        newNode.next = head;
        head = newNode;
        size++;
    }
    
    public void addLast(T data) {
        Node newNode = new Node(data);
        if (head == null) {
            head = newNode;
        } else {
            Node current = head;
            while (current.next != null) {
                current = current.next;
            }
            current.next = newNode;
        }
        size++;
    }
    
    public T removeFirst() {
        if (head == null) return null;
        T data = head.data;
        head = head.next;
        size--;
        return data;
    }
    
    // IMPORTANT INTERVIEW METHODS:
    
    // 1. Reverse LinkedList (MUST KNOW!)
    public void reverse() {
        Node prev = null;
        Node current = head;
        Node next = null;
        
        while (current != null) {
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        head = prev;
    }
    
    // 2. Find Middle Element
    public T findMiddle() {
        if (head == null) return null;
        
        Node slow = head;
        Node fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow.data;
    }
    
    // 3. Detect Cycle (Floyd's Algorithm)
    public boolean hasCycle() {
        if (head == null) return false;
        
        Node slow = head;
        Node fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) return true;
        }
        return false;
    }
    
    // 4. Remove Nth Node from End
    public void removeNthFromEnd(int n) {
        Node dummy = new Node(null);
        dummy.next = head;
        Node first = dummy;
        Node second = dummy;
        
        // Move first n+1 steps ahead
        for (int i = 0; i <= n; i++) {
            first = first.next;
        }
        
        // Move both pointers until first reaches end
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        
        // Remove the node
        second.next = second.next.next;
        head = dummy.next;
    }
}
```

#### ⏰ Session 2 (1.5 Hours): Problem Solving Sprint

**Solve These 5 Problems (18 min each):**

1. **[LeetCode #206: Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)**
   - Both iterative and recursive solutions

2. **[LeetCode #21: Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)**
   ```java
   public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
       if (l1 == null) return l2;
       if (l2 == null) return l1;
       
       if (l1.val < l2.val) {
           l1.next = mergeTwoLists(l1.next, l2);
           return l1;
       } else {
           l2.next = mergeTwoLists(l1, l2.next);
           return l2;
       }
   }
   ```

3. **[LeetCode #141: Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)**

4. **[LeetCode #876: Middle of Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)**

5. **[LeetCode #83: Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)**

#### 📝 Saturday Checklist:
- [ ] Implement complete LinkedList class
- [ ] Master reverse, middle, cycle detection
- [ ] Solve all 5 LeetCode problems
- [ ] Time yourself for each problem
- [ ] Review solutions you couldn't solve

---

### 📆 Sunday (2-3 Hours)
**Topic: Stack, Queue Implementation + Weekly Contest**

#### ⏰ Session 1 (1.5 Hours): Stack & Queue

**0-30 min: Learn Concepts**
- 📹 [Stack Data Structure](https://www.youtube.com/watch?v=I5lq6sCuABE) (15 min)
- 📹 [Queue Data Structure](https://www.youtube.com/watch?v=PvWxOcWDXAw) (15 min)

**30-90 min: Implementations**
```java
// 1. Stack using Array
class MyStack {
    private int[] arr;
    private int top;
    private int capacity;
    
    public MyStack(int size) {
        arr = new int[size];
        capacity = size;
        top = -1;
    }
    
    public void push(int x) {
        if (isFull()) {
            throw new StackOverflowError("Stack is full");
        }
        arr[++top] = x;
    }
    
    public int pop() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return arr[top--];
    }
    
    public int peek() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return arr[top];
    }
    
    public boolean isEmpty() {
        return top == -1;
    }
    
    public boolean isFull() {
        return top == capacity - 1;
    }
}

// 2. Queue using LinkedList
class MyQueue {
    private Node front, rear;
    private int size;
    
    class Node {
        int data;
        Node next;
        Node(int data) {
            this.data = data;
        }
    }
    
    public void enqueue(int data) {
        Node newNode = new Node(data);
        if (rear == null) {
            front = rear = newNode;
        } else {
            rear.next = newNode;
            rear = newNode;
        }
        size++;
    }
    
    public int dequeue() {
        if (front == null) {
            throw new NoSuchElementException();
        }
        int data = front.data;
        front = front.next;
        if (front == null) {
            rear = null;
        }
        size--;
        return data;
    }
}

// 3. IMPORTANT: Implement Queue using Stacks (Interview Favorite!)
class QueueUsingStacks {
    Stack<Integer> stack1 = new Stack<>();
    Stack<Integer> stack2 = new Stack<>();
    
    public void enqueue(int x) {
        stack1.push(x);
    }
    
    public int dequeue() {
        if (stack2.isEmpty()) {
            while (!stack1.isEmpty()) {
                stack2.push(stack1.pop());
            }
        }
        if (stack2.isEmpty()) {
            throw new NoSuchElementException();
        }
        return stack2.pop();
    }
}
```

#### ⏰ Session 2 (1.5 Hours): Problems + Review

**0-60 min: Solve Stack/Queue Problems**

1. **[LeetCode #20: Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)** (MUST KNOW!)
   ```java
   public boolean isValid(String s) {
       Stack<Character> stack = new Stack<>();
       
       for (char c : s.toCharArray()) {
           if (c == '(' || c == '{' || c == '[') {
               stack.push(c);
           } else {
               if (stack.isEmpty()) return false;
               
               char top = stack.pop();
               if ((c == ')' && top != '(') ||
                   (c == '}' && top != '{') ||
                   (c == ']' && top != '[')) {
                   return false;
               }
           }
       }
       return stack.isEmpty();
   }
   ```

2. **[LeetCode #155: Min Stack](https://leetcode.com/problems/min-stack/)**

3. **[LeetCode #232: Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)**

**60-90 min: Week 1 Complete Review**
- Review all implementations
- Create a summary sheet with:
  - Time complexities for all operations
  - When to use each data structure
  - Common interview patterns
- Plan for Week 2

#### 📝 Sunday Checklist:
- [ ] Implement Stack and Queue from scratch
- [ ] Solve Valid Parentheses problem
- [ ] Complete at least 3 stack/queue problems
- [ ] Create Week 1 summary notes
- [ ] Prepare for Week 2

---

## 📊 Week 1 Progress Tracker

### Problems Solved:
| Day | Easy | Medium | Hard | Total |
|-----|------|--------|------|-------|
| Mon | 0 | 0 | 0 | 0 |
| Tue | 2 | 0 | 0 | 2 |
| Wed | 1 | 0 | 0 | 1 |
| Thu | 1 | 0 | 0 | 1 |
| Fri | 2 | 0 | 0 | 2 |
| Sat | 5 | 0 | 0 | 5 |
| Sun | 3 | 0 | 0 | 3 |
| **Total** | **14** | **0** | **0** | **14** |

### Concepts Mastered:
- ✅ ArrayList internals
- ✅ HashMap working
- ✅ LinkedList operations
- ✅ Stack & Queue
- ✅ Thread-safe collections

### Interview Patterns Learned:
1. Two Pointer technique
2. HashMap for optimization
3. Fast & Slow pointers
4. Stack for matching problems
5. Queue using stacks

---

## 🎯 Week 1 Key Takeaways

### Must-Remember for Interviews:
1. **HashMap**: Default capacity 16, load factor 0.75, collision by chaining
2. **ArrayList**: 50% capacity increase on resize (not doubling!)
3. **LinkedList**: Always consider dummy node for edge cases
4. **Time Complexities**: 
   - ArrayList: Access O(1), Insert/Delete O(n)
   - LinkedList: Access O(n), Insert/Delete at head O(1)
   - HashMap: Average O(1), Worst O(n)
5. **Thread Safety**: Use ConcurrentHashMap, not Collections.synchronizedMap()

### Common Mistakes to Avoid:
❌ Forgetting null checks in LinkedList
❌ Not handling HashMap collisions
❌ Using HashMap when order matters (use LinkedHashMap)
❌ Not considering thread safety in concurrent scenarios

---

## 🚀 Ready for Week 2?

### Week 2 Preview:
- Binary Search Deep Dive
- Recursion Fundamentals
- Tree Basics (Binary Tree, BST)
- 20+ more problems
- First mock interview (Saturday)

### Success Tips:
1. **Code daily** - Even 30 minutes counts
2. **Debug your code** - Don't just run, understand
3. **Time yourself** - Speed matters in interviews
4. **Ask why** - Understand the reasoning
5. **Stay consistent** - Trust the process!

---

## 📱 Quick Reference Card

### Daily Shortcuts:
```bash
# Terminal Commands
cd ~/FAANG_Prep/Week1
javac MyHashMap.java && java MyHashMap
git add . && git commit -m "Day X complete"

# VS Code Shortcuts
Ctrl+Shift+P : Command palette
Ctrl+` : Terminal
Ctrl+B : Toggle sidebar
Alt+Shift+F : Format code
```

### Resources Bookmarks:
1. [LeetCode Profile](https://leetcode.com)
2. [NeetCode Videos](https://www.youtube.com/@NeetCode)
3. [Java Documentation](https://docs.oracle.com/javase/8/docs/api/)
4. [Big-O Cheat Sheet](https://www.bigocheatsheet.com/)

---

## 💪 You've Got This!

Remember: **Every expert was once a beginner.** This first week lays the foundation for your entire journey. Focus on understanding, not memorization. The patterns you learn this week will appear throughout your preparation.

**End of Week 1 Reward**: Solve your first 14 problems! 🎉

---

*Week 1 Schedule Version 1.0*
*Time Commitment: 11 hours*
*Problems Target: 14*
*Next Week: Binary Search & Recursion*