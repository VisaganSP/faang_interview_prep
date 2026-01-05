# 🚀 FAANG Interview Masterguide: Advanced Java & Python

> **Your path to becoming a God-Tier Software Engineer**

---

## 📑 Table of Contents

### Part 1: Advanced Java
- [1.1 JVM Internals & Architecture](#11-jvm-internals--architecture)
- [1.2 Memory Management & Garbage Collection](#12-memory-management--garbage-collection)
- [1.3 Multithreading & Concurrency](#13-multithreading--concurrency)
- [1.4 Advanced Generics](#14-advanced-generics)
- [1.5 Collections Framework Deep Dive](#15-collections-framework-deep-dive)
- [1.6 Streams & Functional Programming](#16-streams--functional-programming)
- [1.7 Reflection API](#17-reflection-api)
- [1.8 Annotations & Annotation Processing](#18-annotations--annotation-processing)
- [1.9 Class Loaders](#19-class-loaders)
- [1.10 Java NIO & Asynchronous I/O](#110-java-nio--asynchronous-io)
- [1.11 Java Memory Model (JMM)](#111-java-memory-model-jmm)

### Part 2: Advanced Python
- [2.1 Python Internals & CPython](#21-python-internals--cpython)
- [2.2 Memory Management in Python](#22-memory-management-in-python)
- [2.3 GIL (Global Interpreter Lock)](#23-gil-global-interpreter-lock)
- [2.4 Decorators (Deep Dive)](#24-decorators-deep-dive)
- [2.5 Generators & Iterators](#25-generators--iterators)
- [2.6 Context Managers](#26-context-managers)
- [2.7 Metaclasses](#27-metaclasses)
- [2.8 Descriptors](#28-descriptors)
- [2.9 Concurrency: asyncio, threading, multiprocessing](#29-concurrency-asyncio-threading-multiprocessing)
- [2.10 Type Hints & Static Typing](#210-type-hints--static-typing)
- [2.11 Magic Methods (Dunder Methods)](#211-magic-methods-dunder-methods)

### Part 3: Design Patterns
- [3.1 Creational Patterns](#31-creational-patterns)
- [3.2 Structural Patterns](#32-structural-patterns)
- [3.3 Behavioral Patterns](#33-behavioral-patterns)
- [3.4 Concurrency Patterns](#34-concurrency-patterns)
- [3.5 Architectural Patterns](#35-architectural-patterns)

### Part 4: FAANG-Specific Topics
- [4.1 System Design Fundamentals](#41-system-design-fundamentals)
- [4.2 Distributed Systems Concepts](#42-distributed-systems-concepts)
- [4.3 Code Quality & Best Practices](#43-code-quality--best-practices)

### Part 5: Resources
- [5.1 YouTube Channels & Playlists](#51-youtube-channels--playlists)
- [5.2 Books](#52-books)
- [5.3 Practice Platforms](#53-practice-platforms)

---

# Part 1: Advanced Java

## 1.1 JVM Internals & Architecture

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                         JVM Architecture                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Class Loader │  │ Runtime Data │  │ Execution Engine │   │
│  │   Subsystem  │  │    Areas     │  │                  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│         │                 │                   │              │
│         ▼                 ▼                   ▼              │
│  ┌─────────────┐   ┌───────────┐      ┌─────────────┐       │
│  │ Bootstrap   │   │ Method    │      │ Interpreter │       │
│  │ Extension   │   │ Area      │      │ JIT Compiler│       │
│  │ Application │   │ Heap      │      │ GC          │       │
│  └─────────────┘   │ Stack     │      └─────────────┘       │
│                    │ PC Reg    │                            │
│                    │ Native    │                            │
│                    └───────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### Topics to Master

| Topic | Description | Difficulty |
|-------|-------------|------------|
| Class Loading Mechanism | Bootstrap, Extension, Application loaders | ⭐⭐⭐ |
| Bytecode Verification | Security & type safety checks | ⭐⭐⭐⭐ |
| JIT Compilation | HotSpot, C1/C2 compilers, tiered compilation | ⭐⭐⭐⭐⭐ |
| Method Area/Metaspace | Class metadata storage (Java 8+) | ⭐⭐⭐ |
| Stack Frame Structure | Local variables, operand stack, frame data | ⭐⭐⭐⭐ |

### Interview Questions
- [ ] Explain the difference between JDK, JRE, and JVM
- [ ] How does JIT compilation work?
- [ ] What is the difference between Method Area and Metaspace?
- [ ] Explain class loading delegation model

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 1.2 Memory Management & Garbage Collection

### Heap Structure (G1 GC)

```
┌─────────────────────────────────────────────────────────────┐
│                          HEAP                                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Young Generation                        │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │    │
│  │  │  Eden   │  │Survivor │  │Survivor │             │    │
│  │  │         │  │   S0    │  │   S1    │             │    │
│  │  └─────────┘  └─────────┘  └─────────┘             │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Old Generation (Tenured)                │    │
│  │                                                      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Garbage Collectors

| GC Type | Use Case | Java Version |
|---------|----------|--------------|
| Serial GC | Single-threaded, small heaps | All |
| Parallel GC | Throughput-focused | Default until Java 8 |
| CMS (Concurrent Mark Sweep) | Low latency | Deprecated Java 9 |
| G1 (Garbage First) | Balanced, large heaps | Default Java 9+ |
| ZGC | Ultra-low latency (<10ms) | Java 11+ |
| Shenandoah | Low pause times | Java 12+ |

### Key Concepts

```java
// Example: Understanding Object Lifecycle
public class MemoryDemo {
    
    // Strong Reference - default, prevents GC
    Object strongRef = new Object();
    
    // Weak Reference - collected when no strong refs exist
    WeakReference<Object> weakRef = new WeakReference<>(new Object());
    
    // Soft Reference - collected when memory is low
    SoftReference<Object> softRef = new SoftReference<>(new Object());
    
    // Phantom Reference - for cleanup actions
    PhantomReference<Object> phantomRef = new PhantomReference<>(
        new Object(), new ReferenceQueue<>()
    );
}
```

### JVM Tuning Parameters

```bash
# Heap Size
-Xms512m          # Initial heap size
-Xmx2g            # Maximum heap size

# GC Selection
-XX:+UseG1GC      # Use G1 Garbage Collector
-XX:+UseZGC       # Use ZGC (Java 11+)

# GC Tuning
-XX:MaxGCPauseMillis=200    # Target max pause time
-XX:G1HeapRegionSize=16m    # G1 region size

# Monitoring
-XX:+PrintGCDetails         # Print GC details
-Xlog:gc*                   # Java 9+ unified logging
```

### Interview Questions
- [ ] Explain the difference between Minor GC and Major GC
- [ ] How does G1 GC achieve low pause times?
- [ ] What is a memory leak in Java and how to detect it?
- [ ] Explain Strong, Weak, Soft, and Phantom references
- [ ] How would you tune JVM for a high-throughput application?

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 1.3 Multithreading & Concurrency

### Thread Lifecycle

```
        ┌──────────────┐
        │     NEW      │
        └──────┬───────┘
               │ start()
               ▼
        ┌──────────────┐
   ┌───▶│   RUNNABLE   │◀──────────────┐
   │    └──────┬───────┘               │
   │           │                       │
   │    ┌──────┴───────┐               │
   │    ▼              ▼               │
┌──┴────────┐   ┌──────────────┐       │
│  BLOCKED  │   │   WAITING    │       │
└───────────┘   │ TIMED_WAITING│       │
                └──────┬───────┘       │
                       │ notify/timeout│
                       └───────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  TERMINATED  │
                       └──────────────┘
```

### Core Concurrency Classes

```java
// 1. ExecutorService - Thread Pool Management
ExecutorService executor = Executors.newFixedThreadPool(4);
Future<String> future = executor.submit(() -> "Result");

// 2. CompletableFuture - Async Programming
CompletableFuture.supplyAsync(() -> fetchData())
    .thenApply(data -> process(data))
    .thenAccept(result -> save(result))
    .exceptionally(ex -> handleError(ex));

// 3. Locks - Fine-grained control
ReentrantLock lock = new ReentrantLock();
ReentrantReadWriteLock rwLock = new ReentrantReadWriteLock();
StampedLock stampedLock = new StampedLock(); // Java 8+

// 4. Atomic Classes - Lock-free thread safety
AtomicInteger counter = new AtomicInteger(0);
AtomicReference<Node> head = new AtomicReference<>();
LongAdder adder = new LongAdder(); // Better for high contention

// 5. Concurrent Collections
ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
CopyOnWriteArrayList<String> list = new CopyOnWriteArrayList<>();
BlockingQueue<Task> queue = new LinkedBlockingQueue<>();
```

### Synchronization Mechanisms

| Mechanism | Use Case | Performance |
|-----------|----------|-------------|
| `synchronized` | Simple mutual exclusion | Moderate |
| `ReentrantLock` | Need tryLock, interruptible | Good |
| `ReadWriteLock` | Read-heavy workloads | Very Good |
| `StampedLock` | Optimistic reads | Excellent |
| `Semaphore` | Limit concurrent access | Good |
| `CountDownLatch` | Wait for N events | Good |
| `CyclicBarrier` | Synchronize N threads | Good |
| `Phaser` | Dynamic thread coordination | Good |

### Advanced Example: Thread-Safe Singleton

```java
// Double-Checked Locking with volatile
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

// Initialization-on-demand holder idiom (Preferred)
public class SingletonBetter {
    private SingletonBetter() {}
    
    private static class Holder {
        static final SingletonBetter INSTANCE = new SingletonBetter();
    }
    
    public static SingletonBetter getInstance() {
        return Holder.INSTANCE;
    }
}
```

### Interview Questions
- [ ] What is the difference between `synchronized` and `ReentrantLock`?
- [ ] Explain the Java Memory Model and happens-before relationship
- [ ] How does ConcurrentHashMap achieve thread safety?
- [ ] What is a deadlock? How to prevent and detect it?
- [ ] Explain Fork/Join framework
- [ ] What is the difference between `volatile` and `synchronized`?

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 1.4 Advanced Generics

### Type Erasure

```java
// At compile time
List<String> strings = new ArrayList<String>();
List<Integer> integers = new ArrayList<Integer>();

// At runtime (after type erasure)
List strings = new ArrayList();  // Same bytecode!
List integers = new ArrayList();
```

### Wildcards & Bounds

```java
// Upper Bounded Wildcard - "Producer Extends"
public void processNumbers(List<? extends Number> numbers) {
    for (Number n : numbers) {
        System.out.println(n.doubleValue());
    }
    // numbers.add(1); // COMPILE ERROR - can only read
}

// Lower Bounded Wildcard - "Consumer Super"
public void addIntegers(List<? super Integer> list) {
    list.add(1);
    list.add(2);
    // Integer i = list.get(0); // COMPILE ERROR - can only write
}

// PECS: Producer Extends, Consumer Super
public static <T> void copy(List<? extends T> src, List<? super T> dest) {
    for (T item : src) {
        dest.add(item);
    }
}
```

### Generic Methods & Type Inference

```java
// Generic method with multiple type parameters
public static <K, V> Map<K, V> zipToMap(List<K> keys, List<V> values) {
    Map<K, V> result = new HashMap<>();
    for (int i = 0; i < keys.size(); i++) {
        result.put(keys.get(i), values.get(i));
    }
    return result;
}

// Self-referential generics (Curiously Recurring Template Pattern)
public abstract class Builder<T extends Builder<T>> {
    public abstract T self();
    
    public T withName(String name) {
        // set name
        return self();
    }
}

// Recursive Type Bounds
public static <T extends Comparable<T>> T max(List<T> list) {
    return Collections.max(list);
}
```

### Type Tokens (Super Type Tokens)

```java
// Problem: Type erasure loses generic info
// Solution: Capture type at class level

public abstract class TypeReference<T> {
    private final Type type;
    
    protected TypeReference() {
        Type superclass = getClass().getGenericSuperclass();
        this.type = ((ParameterizedType) superclass).getActualTypeArguments()[0];
    }
    
    public Type getType() { return type; }
}

// Usage
TypeReference<List<String>> typeRef = new TypeReference<List<String>>() {};
System.out.println(typeRef.getType()); // java.util.List<java.lang.String>
```

### Interview Questions
- [ ] What is type erasure and why does Java use it?
- [ ] Explain PECS principle
- [ ] What is the difference between `List<Object>` and `List<?>`?
- [ ] How to create a generic array in Java?
- [ ] Explain covariance and contravariance in generics

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 1.5 Collections Framework Deep Dive

### Collections Hierarchy

```
                     Iterable
                        │
                    Collection
           ┌────────────┼────────────┐
           │            │            │
          List         Set         Queue
           │            │            │
    ┌──────┴──────┐  ┌──┴───┐   ┌───┴────┐
ArrayList  LinkedList  HashSet  PriorityQueue
    │                TreeSet   ArrayDeque
 Vector              LinkedHashSet
    │
  Stack
```

### Internal Implementations

| Collection | Data Structure | Time Complexity |
|------------|----------------|-----------------|
| ArrayList | Dynamic Array | O(1) get, O(n) insert |
| LinkedList | Doubly Linked List | O(n) get, O(1) insert at ends |
| HashMap | Hash Table + Red-Black Tree | O(1) avg, O(log n) worst |
| TreeMap | Red-Black Tree | O(log n) all operations |
| LinkedHashMap | Hash Table + Linked List | O(1) + maintains order |
| ConcurrentHashMap | Segmented Hash Table | O(1) thread-safe |
| PriorityQueue | Binary Heap | O(log n) insert/remove |

### HashMap Internals (Java 8+)

```java
// HashMap uses array of Nodes (buckets)
// When bucket has > 8 elements, converts to Red-Black Tree
// When bucket has < 6 elements, converts back to LinkedList

// Key methods to understand:
// 1. hash() - Spreads higher bits of hashCode
static final int hash(Object key) {
    int h;
    return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
}

// 2. Bucket index calculation
int index = (n - 1) & hash;  // n = table length (power of 2)

// 3. Load factor and resizing
// Default: capacity=16, loadFactor=0.75
// Resize when: size > capacity * loadFactor
```

### Custom Comparator & Comparable

```java
// Comparable - Natural ordering (within class)
public class Employee implements Comparable<Employee> {
    private String name;
    private int salary;
    
    @Override
    public int compareTo(Employee other) {
        return Integer.compare(this.salary, other.salary);
    }
}

// Comparator - External ordering (flexible)
Comparator<Employee> byName = Comparator.comparing(Employee::getName);
Comparator<Employee> bySalaryDesc = Comparator.comparingInt(Employee::getSalary).reversed();
Comparator<Employee> complex = Comparator
    .comparing(Employee::getDepartment)
    .thenComparing(Employee::getSalary)
    .thenComparing(Employee::getName);
```

### Interview Questions
- [ ] How does HashMap handle collisions?
- [ ] Why is the initial capacity of HashMap a power of 2?
- [ ] When does HashMap convert bucket to Red-Black Tree?
- [ ] Difference between HashMap and ConcurrentHashMap
- [ ] How does TreeMap maintain sorted order?
- [ ] What happens if you modify an object used as key in HashMap?

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 1.6 Streams & Functional Programming

### Stream Pipeline

```
┌──────────┐    ┌────────────────┐    ┌──────────┐
│  Source  │───▶│ Intermediate   │───▶│ Terminal │
│          │    │  Operations    │    │Operation │
└──────────┘    └────────────────┘    └──────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
      filter()     map()      sorted()
      distinct()   flatMap()  peek()
      limit()      skip()     
```

### Advanced Stream Operations

```java
// 1. FlatMap - Flatten nested structures
List<List<Integer>> nested = Arrays.asList(
    Arrays.asList(1, 2), 
    Arrays.asList(3, 4)
);
List<Integer> flat = nested.stream()
    .flatMap(Collection::stream)
    .collect(Collectors.toList()); // [1, 2, 3, 4]

// 2. Collectors - Advanced collection
Map<String, List<Employee>> byDept = employees.stream()
    .collect(Collectors.groupingBy(Employee::getDepartment));

Map<String, Double> avgSalaryByDept = employees.stream()
    .collect(Collectors.groupingBy(
        Employee::getDepartment,
        Collectors.averagingDouble(Employee::getSalary)
    ));

// 3. Reduce - Custom aggregation
int sum = numbers.stream()
    .reduce(0, (a, b) -> a + b);

Optional<Integer> max = numbers.stream()
    .reduce(Integer::max);

// 4. Parallel Streams
long count = data.parallelStream()
    .filter(x -> x > 0)
    .count();

// 5. Custom Collector
Collector<String, StringBuilder, String> joiner = Collector.of(
    StringBuilder::new,           // Supplier
    StringBuilder::append,        // Accumulator
    StringBuilder::append,        // Combiner
    StringBuilder::toString       // Finisher
);
```

### Functional Interfaces

```java
// Core functional interfaces
Function<T, R>      // T -> R
BiFunction<T, U, R> // (T, U) -> R
Consumer<T>         // T -> void
Supplier<T>         // () -> T
Predicate<T>        // T -> boolean
UnaryOperator<T>    // T -> T
BinaryOperator<T>   // (T, T) -> T

// Method references
list.forEach(System.out::println);     // Instance method
strings.map(String::toUpperCase);       // Instance method on element
numbers.map(Integer::parseInt);         // Static method
list.map(Employee::new);                // Constructor
```

### Interview Questions
- [ ] What is lazy evaluation in streams?
- [ ] When to use parallel streams and when not to?
- [ ] Difference between `map()` and `flatMap()`
- [ ] How does `reduce()` work?
- [ ] What are stateful vs stateless intermediate operations?

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 1.7 Reflection API

### Core Reflection Operations

```java
// 1. Get Class object
Class<?> clazz = MyClass.class;
Class<?> clazz = obj.getClass();
Class<?> clazz = Class.forName("com.example.MyClass");

// 2. Inspect class members
Field[] fields = clazz.getDeclaredFields();
Method[] methods = clazz.getDeclaredMethods();
Constructor<?>[] constructors = clazz.getDeclaredConstructors();

// 3. Access private members
Field privateField = clazz.getDeclaredField("secretField");
privateField.setAccessible(true);
Object value = privateField.get(instance);

// 4. Invoke methods dynamically
Method method = clazz.getMethod("doSomething", String.class);
Object result = method.invoke(instance, "argument");

// 5. Create instances
Constructor<?> constructor = clazz.getConstructor(String.class);
Object newInstance = constructor.newInstance("arg");
```

### Practical Example: Simple Dependency Injection

```java
public class SimpleInjector {
    private Map<Class<?>, Object> instances = new HashMap<>();
    
    public <T> T getInstance(Class<T> clazz) throws Exception {
        if (!instances.containsKey(clazz)) {
            T instance = createInstance(clazz);
            instances.put(clazz, instance);
        }
        return clazz.cast(instances.get(clazz));
    }
    
    private <T> T createInstance(Class<T> clazz) throws Exception {
        Constructor<T> constructor = clazz.getDeclaredConstructor();
        T instance = constructor.newInstance();
        
        // Inject @Autowired fields
        for (Field field : clazz.getDeclaredFields()) {
            if (field.isAnnotationPresent(Autowired.class)) {
                field.setAccessible(true);
                Object dependency = getInstance(field.getType());
                field.set(instance, dependency);
            }
        }
        return instance;
    }
}
```

### Interview Questions
- [ ] What are the performance implications of reflection?
- [ ] How to access private fields using reflection?
- [ ] What is the difference between `getMethod()` and `getDeclaredMethod()`?
- [ ] How do frameworks like Spring use reflection?

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 1.8 Annotations & Annotation Processing

### Custom Annotations

```java
// Define annotation
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface Cached {
    int ttlSeconds() default 300;
    String key() default "";
}

// Use annotation
@Cached(ttlSeconds = 60, key = "users")
public List<User> getUsers() {
    return repository.findAll();
}

// Process annotation at runtime
Method method = clazz.getMethod("getUsers");
if (method.isAnnotationPresent(Cached.class)) {
    Cached cached = method.getAnnotation(Cached.class);
    int ttl = cached.ttlSeconds();
    String key = cached.key();
}
```

### Annotation Processing (Compile-time)

```java
// Processor for compile-time annotation processing
@SupportedAnnotationTypes("com.example.Builder")
@SupportedSourceVersion(SourceVersion.RELEASE_11)
public class BuilderProcessor extends AbstractProcessor {
    
    @Override
    public boolean process(Set<? extends TypeElement> annotations, 
                          RoundEnvironment roundEnv) {
        for (Element element : roundEnv.getElementsAnnotatedWith(Builder.class)) {
            // Generate builder class code
            generateBuilderClass(element);
        }
        return true;
    }
}
```

### Common Built-in Annotations

| Annotation | Purpose |
|------------|---------|
| `@Override` | Compile-time check for method override |
| `@Deprecated` | Mark as deprecated |
| `@SuppressWarnings` | Suppress compiler warnings |
| `@FunctionalInterface` | Ensure single abstract method |
| `@SafeVarargs` | Suppress heap pollution warnings |

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 1.9 Class Loaders

### Class Loader Hierarchy

```
┌─────────────────────────────────────┐
│       Bootstrap ClassLoader         │  ← JRE/lib (rt.jar, core classes)
│         (Native code)               │
└─────────────────┬───────────────────┘
                  │ parent
┌─────────────────▼───────────────────┐
│       Extension ClassLoader         │  ← JRE/lib/ext
│    (sun.misc.Launcher$ExtClassLoader)│
└─────────────────┬───────────────────┘
                  │ parent
┌─────────────────▼───────────────────┐
│      Application ClassLoader        │  ← Classpath
│   (sun.misc.Launcher$AppClassLoader)│
└─────────────────┬───────────────────┘
                  │ parent
┌─────────────────▼───────────────────┐
│       Custom ClassLoaders           │  ← Your custom loaders
└─────────────────────────────────────┘
```

### Custom ClassLoader Example

```java
public class HotSwapClassLoader extends ClassLoader {
    private String classPath;
    
    public HotSwapClassLoader(String classPath, ClassLoader parent) {
        super(parent);
        this.classPath = classPath;
    }
    
    @Override
    protected Class<?> findClass(String name) throws ClassNotFoundException {
        byte[] classData = loadClassData(name);
        if (classData == null) {
            throw new ClassNotFoundException(name);
        }
        return defineClass(name, classData, 0, classData.length);
    }
    
    private byte[] loadClassData(String className) {
        String path = classPath + "/" + className.replace('.', '/') + ".class";
        try (InputStream is = new FileInputStream(path)) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buffer = new byte[1024];
            int len;
            while ((len = is.read(buffer)) != -1) {
                baos.write(buffer, 0, len);
            }
            return baos.toByteArray();
        } catch (IOException e) {
            return null;
        }
    }
}
```

### Interview Questions
- [ ] Explain the delegation model in class loading
- [ ] How to implement hot class reloading?
- [ ] What is the difference between `loadClass()` and `findClass()`?
- [ ] How do application servers achieve class isolation?

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 1.10 Java NIO & Asynchronous I/O

### NIO Components

```java
// 1. Buffers - Data containers
ByteBuffer buffer = ByteBuffer.allocate(1024);
buffer.put(data);
buffer.flip();  // Switch to read mode
while (buffer.hasRemaining()) {
    channel.write(buffer);
}
buffer.clear();  // Switch back to write mode

// 2. Channels - Bidirectional I/O
FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ);
SocketChannel socketChannel = SocketChannel.open();

// 3. Selectors - Multiplexed I/O
Selector selector = Selector.open();
serverChannel.register(selector, SelectionKey.OP_ACCEPT);

while (true) {
    selector.select();  // Blocks until events
    Set<SelectionKey> keys = selector.selectedKeys();
    for (SelectionKey key : keys) {
        if (key.isAcceptable()) {
            // Handle new connection
        } else if (key.isReadable()) {
            // Handle read
        }
    }
}
```

### Asynchronous I/O (NIO.2)

```java
// AsynchronousFileChannel
AsynchronousFileChannel channel = AsynchronousFileChannel.open(path);

// Callback-based
channel.read(buffer, 0, null, new CompletionHandler<Integer, Void>() {
    @Override
    public void completed(Integer result, Void attachment) {
        System.out.println("Read " + result + " bytes");
    }
    
    @Override
    public void failed(Throwable exc, Void attachment) {
        exc.printStackTrace();
    }
});

// Future-based
Future<Integer> future = channel.read(buffer, 0);
Integer bytesRead = future.get();  // Blocks
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 1.11 Java Memory Model (JMM)

### Happens-Before Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                  Happens-Before Rules                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Program Order: Each action in a thread happens-before    │
│    every subsequent action in that thread                   │
│                                                             │
│ 2. Monitor Lock: An unlock on a monitor happens-before      │
│    every subsequent lock on that monitor                    │
│                                                             │
│ 3. Volatile: A write to a volatile field happens-before     │
│    every subsequent read of that field                      │
│                                                             │
│ 4. Thread Start: A call to Thread.start() happens-before    │
│    any action in the started thread                         │
│                                                             │
│ 5. Thread Join: All actions in a thread happen-before       │
│    any other thread returns from a join() on that thread    │
└─────────────────────────────────────────────────────────────┘
```

### Volatile vs Synchronized

```java
// Volatile - Visibility guarantee only
private volatile boolean flag = false;

// Thread 1
flag = true;  // Write is visible to all threads immediately

// Thread 2
if (flag) {
    // Guaranteed to see the update
}

// Synchronized - Visibility + Atomicity
private int count = 0;

public synchronized void increment() {
    count++;  // Atomic: read-modify-write
}
```

### Memory Barriers

```java
// Full fence (LoadStore + StoreLoad + LoadLoad + StoreStore)
// synchronized provides full fence

// Acquire fence (LoadLoad + LoadStore)
// Reading volatile provides acquire semantics

// Release fence (StoreStore + LoadStore)  
// Writing volatile provides release semantics
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

# Part 2: Advanced Python

## 2.1 Python Internals & CPython

### CPython Object Model

```
┌──────────────────────────────────────────────┐
│              PyObject (base)                  │
├──────────────────────────────────────────────┤
│  ob_refcnt  (reference count)                │
│  ob_type    (pointer to type object)         │
└──────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│    PyIntObject   │    │   PyListObject   │
├──────────────────┤    ├──────────────────┤
│  ob_refcnt       │    │  ob_refcnt       │
│  ob_type         │    │  ob_type         │
│  ob_ival (value) │    │  ob_size         │
└──────────────────┘    │  ob_item (array) │
                        └──────────────────┘
```

### Everything is an Object

```python
# Even functions and classes are objects
def my_func():
    pass

print(type(my_func))  # <class 'function'>
print(my_func.__class__.__bases__)  # (<class 'object'>,)

# Classes are instances of 'type'
class MyClass:
    pass

print(type(MyClass))  # <class 'type'>
print(type(type))     # <class 'type'> - type is its own type!
```

### Name Resolution (LEGB Rule)

```python
# L - Local
# E - Enclosing (closure)
# G - Global
# B - Built-in

x = "global"

def outer():
    x = "enclosing"
    
    def inner():
        x = "local"
        print(x)  # "local"
    
    inner()
    print(x)  # "enclosing"

outer()
print(x)  # "global"
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 2.2 Memory Management in Python

### Reference Counting + Garbage Collection

```python
import sys
import gc

# Reference counting
a = [1, 2, 3]
print(sys.getrefcount(a))  # 2 (a + function argument)

b = a
print(sys.getrefcount(a))  # 3

del b
print(sys.getrefcount(a))  # 2

# Garbage collection for cycles
class Node:
    def __init__(self):
        self.next = None

# Create a cycle
n1 = Node()
n2 = Node()
n1.next = n2
n2.next = n1  # Cycle!

# Reference counting can't handle this
del n1, n2  # Objects still exist due to cycle

# GC handles cycles
gc.collect()  # Cleans up circular references
```

### Memory Optimization

```python
# __slots__ - Reduce memory footprint
class RegularClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SlottedClass:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# SlottedClass uses ~40% less memory per instance

# Interning - Reuse immutable objects
a = "hello"
b = "hello"
print(a is b)  # True - same object (interned)

# Small integers are cached (-5 to 256)
a = 256
b = 256
print(a is b)  # True

a = 257
b = 257
print(a is b)  # False (may vary by implementation)
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 2.3 GIL (Global Interpreter Lock)

### What is GIL?

```
┌─────────────────────────────────────────────────────────────┐
│                         GIL                                  │
├─────────────────────────────────────────────────────────────┤
│  • Mutex that protects access to Python objects             │
│  • Only one thread can execute Python bytecode at a time    │
│  • Released during I/O operations                           │
│  • Impacts CPU-bound multi-threaded programs                │
└─────────────────────────────────────────────────────────────┘

Timeline with GIL:
Thread 1: ████░░░░████░░░░████
Thread 2: ░░░░████░░░░████░░░░
               └── Threads take turns
```

### When GIL Matters

```python
import threading
import time

# CPU-bound task - GIL is a bottleneck
def cpu_bound():
    count = 0
    for i in range(100_000_000):
        count += 1

# Single thread
start = time.time()
cpu_bound()
print(f"Single thread: {time.time() - start:.2f}s")

# Multi-threaded (NOT faster due to GIL!)
start = time.time()
t1 = threading.Thread(target=cpu_bound)
t2 = threading.Thread(target=cpu_bound)
t1.start(); t2.start()
t1.join(); t2.join()
print(f"Two threads: {time.time() - start:.2f}s")  # ~Same or slower!

# Solution: Use multiprocessing for CPU-bound
from multiprocessing import Process

start = time.time()
p1 = Process(target=cpu_bound)
p2 = Process(target=cpu_bound)
p1.start(); p2.start()
p1.join(); p2.join()
print(f"Two processes: {time.time() - start:.2f}s")  # ~2x faster!
```

### When GIL Doesn't Matter

```python
import threading
import requests

# I/O bound - GIL released during I/O
def fetch_url(url):
    return requests.get(url)

# Multi-threading works well for I/O
urls = ["https://api.example.com"] * 10
threads = [threading.Thread(target=fetch_url, args=(url,)) for url in urls]
# This will be faster than sequential!
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 2.4 Decorators (Deep Dive)

### Decorator Fundamentals

```python
# A decorator is a function that takes a function and returns a function
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")

# Equivalent to: say_hello = my_decorator(say_hello)
```

### Preserving Metadata with functools.wraps

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)  # Preserves __name__, __doc__, etc.
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def example():
    """This is the docstring."""
    pass

print(example.__name__)  # "example" (not "wrapper")
print(example.__doc__)   # "This is the docstring."
```

### Decorators with Arguments

```python
from functools import wraps

def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    print(f"Hello, {name}!")

greet("World")  # Prints 3 times
```

### Class-based Decorators

```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call {self.count}")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

say_hello()  # Call 1, Hello!
say_hello()  # Call 2, Hello!
print(say_hello.count)  # 2
```

### Practical Decorators

```python
import time
from functools import wraps, lru_cache

# 1. Timing decorator
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

# 2. Retry decorator
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

# 3. Memoization (built-in)
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 2.5 Generators & Iterators

### Iterator Protocol

```python
class CountDown:
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

for num in CountDown(5):
    print(num)  # 5, 4, 3, 2, 1
```

### Generators (Lazy Iterators)

```python
# Generator function
def countdown(n):
    while n > 0:
        yield n
        n -= 1

# Generator expression
squares = (x**2 for x in range(10))

# Memory efficient - values computed on demand
def read_large_file(file_path):
    with open(file_path) as f:
        for line in f:
            yield line.strip()
```

### Advanced Generator Features

```python
# 1. send() - Two-way communication
def accumulator():
    total = 0
    while True:
        value = yield total
        if value is None:
            break
        total += value

acc = accumulator()
next(acc)       # Initialize (returns 0)
acc.send(10)    # Returns 10
acc.send(20)    # Returns 30
acc.send(5)     # Returns 35

# 2. yield from - Delegate to sub-generator
def chain(*iterables):
    for it in iterables:
        yield from it

list(chain([1, 2], [3, 4]))  # [1, 2, 3, 4]

# 3. Generator-based coroutines (legacy, use async/await now)
def coroutine():
    while True:
        x = yield
        print(f"Received: {x}")
```

### Interview Questions
- [ ] What is the difference between `yield` and `return`?
- [ ] How do generators save memory?
- [ ] Explain `yield from`
- [ ] How does `send()` work with generators?

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 2.6 Context Managers

### The Protocol

```python
class ManagedFile:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, 'r')
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        # Return True to suppress exceptions
        return False

with ManagedFile('test.txt') as f:
    content = f.read()
```

### Using contextlib

```python
from contextlib import contextmanager, suppress, ExitStack

# 1. @contextmanager decorator
@contextmanager
def managed_file(filename):
    f = open(filename, 'r')
    try:
        yield f
    finally:
        f.close()

# 2. suppress - Ignore specific exceptions
with suppress(FileNotFoundError):
    os.remove('nonexistent.txt')

# 3. ExitStack - Dynamic context management
with ExitStack() as stack:
    files = [stack.enter_context(open(f)) for f in filenames]
    # All files will be closed when exiting
```

### Practical Examples

```python
# Database transaction manager
@contextmanager
def transaction(connection):
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise

# Timing context
@contextmanager
def timer(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.4f}s")

with timer("Operation"):
    time.sleep(1)
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 2.7 Metaclasses

### Class Creation Process

```
┌─────────────────────────────────────────────────────────────┐
│                  Class Creation Flow                         │
├─────────────────────────────────────────────────────────────┤
│  1. Python reads class definition                           │
│  2. Determines metaclass (default: type)                    │
│  3. Prepares class namespace (__prepare__)                  │
│  4. Executes class body in namespace                        │
│  5. Calls metaclass to create class object                  │
└─────────────────────────────────────────────────────────────┘
```

### Custom Metaclass

```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        print("Initializing database")

db1 = Database()  # "Initializing database"
db2 = Database()  # No output - same instance
print(db1 is db2)  # True
```

### Metaclass Methods

```python
class MyMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Called when creating the class object
        print(f"Creating class: {name}")
        return super().__new__(mcs, name, bases, namespace)
    
    def __init__(cls, name, bases, namespace):
        # Called after class object is created
        print(f"Initializing class: {name}")
        super().__init__(name, bases, namespace)
    
    def __call__(cls, *args, **kwargs):
        # Called when instantiating the class
        print(f"Creating instance of: {cls.__name__}")
        return super().__call__(*args, **kwargs)

class MyClass(metaclass=MyMeta):
    pass

# Output:
# Creating class: MyClass
# Initializing class: MyClass

obj = MyClass()
# Output:
# Creating instance of: MyClass
```

### Practical Example: Auto-registration

```python
class PluginRegistry(type):
    plugins = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name != 'Plugin':  # Don't register base class
            mcs.plugins[name] = cls
        return cls

class Plugin(metaclass=PluginRegistry):
    pass

class JSONPlugin(Plugin):
    def parse(self, data):
        return json.loads(data)

class XMLPlugin(Plugin):
    def parse(self, data):
        return xml.parse(data)

print(PluginRegistry.plugins)
# {'JSONPlugin': <class 'JSONPlugin'>, 'XMLPlugin': <class 'XMLPlugin'>}
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 2.8 Descriptors

### Descriptor Protocol

```python
class Descriptor:
    def __get__(self, obj, objtype=None):
        # Called when attribute is accessed
        pass
    
    def __set__(self, obj, value):
        # Called when attribute is assigned
        pass
    
    def __delete__(self, obj):
        # Called when attribute is deleted
        pass
```

### Practical Descriptors

```python
# 1. Type-validated attribute
class Typed:
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"Expected {self.expected_type}")
        obj.__dict__[self.name] = value

class Person:
    name = Typed('name', str)
    age = Typed('age', int)
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Alice", 30)  # OK
p.age = "thirty"  # TypeError!

# 2. Lazy property
class LazyProperty:
    def __init__(self, func):
        self.func = func
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self.func(obj)
        setattr(obj, self.func.__name__, value)  # Cache result
        return value

class DataProcessor:
    @LazyProperty
    def expensive_data(self):
        print("Computing...")
        return sum(range(1000000))

dp = DataProcessor()
print(dp.expensive_data)  # Computing... 499999500000
print(dp.expensive_data)  # 499999500000 (cached, no "Computing...")
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 2.9 Concurrency: asyncio, threading, multiprocessing

### Comparison

| Feature | threading | multiprocessing | asyncio |
|---------|-----------|-----------------|---------|
| Parallelism | No (GIL) | Yes | No |
| Best for | I/O-bound | CPU-bound | I/O-bound |
| Memory | Shared | Separate | Shared |
| Overhead | Medium | High | Low |
| Complexity | Medium | Medium | High |

### asyncio Basics

```python
import asyncio

# Coroutine definition
async def fetch_data(url):
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # Simulated I/O
    return f"Data from {url}"

# Running coroutines
async def main():
    # Sequential
    result1 = await fetch_data("url1")
    result2 = await fetch_data("url2")
    
    # Concurrent
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3")
    )
    
    # With timeout
    try:
        result = await asyncio.wait_for(fetch_data("url"), timeout=0.5)
    except asyncio.TimeoutError:
        print("Timeout!")

asyncio.run(main())
```

### Advanced asyncio

```python
import asyncio
from asyncio import Queue

# Producer-Consumer pattern
async def producer(queue: Queue):
    for i in range(5):
        await queue.put(i)
        print(f"Produced: {i}")
        await asyncio.sleep(0.1)
    await queue.put(None)  # Signal completion

async def consumer(queue: Queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")
        await asyncio.sleep(0.2)

async def main():
    queue = Queue()
    await asyncio.gather(producer(queue), consumer(queue))

# Semaphore for rate limiting
async def rate_limited_fetch(url, semaphore):
    async with semaphore:
        return await fetch_data(url)

async def main():
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent
    urls = [f"url{i}" for i in range(100)]
    tasks = [rate_limited_fetch(url, semaphore) for url in urls]
    results = await asyncio.gather(*tasks)
```

### multiprocessing with ProcessPoolExecutor

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# CPU-bound task
def cpu_intensive(n):
    return sum(i * i for i in range(n))

# Using ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(cpu_intensive, [10**6] * 4))

# Shared state with Manager
def worker(shared_dict, key, value):
    shared_dict[key] = value

with mp.Manager() as manager:
    shared_dict = manager.dict()
    processes = [
        mp.Process(target=worker, args=(shared_dict, i, i**2))
        for i in range(4)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print(dict(shared_dict))  # {0: 0, 1: 1, 2: 4, 3: 9}
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 2.10 Type Hints & Static Typing

### Basic Type Hints

```python
from typing import (
    List, Dict, Set, Tuple, Optional, Union, 
    Callable, TypeVar, Generic, Protocol
)

# Basic types
def greet(name: str) -> str:
    return f"Hello, {name}"

# Collections
def process(items: List[int]) -> Dict[str, int]:
    return {"sum": sum(items), "count": len(items)}

# Optional (can be None)
def find_user(id: int) -> Optional[User]:
    return db.get(id)

# Union (multiple types)
def parse(value: Union[str, bytes]) -> dict:
    if isinstance(value, bytes):
        value = value.decode()
    return json.loads(value)

# Callable
def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)
```

### Advanced Type Hints

```python
# TypeVar for generics
T = TypeVar('T')

def first(items: List[T]) -> T:
    return items[0]

# Bounded TypeVar
Numeric = TypeVar('Numeric', int, float)

def add(a: Numeric, b: Numeric) -> Numeric:
    return a + b

# Generic classes
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()

# Protocol (structural subtyping)
class Drawable(Protocol):
    def draw(self) -> None: ...

def render(item: Drawable) -> None:
    item.draw()

# Any class with draw() method works
class Circle:
    def draw(self) -> None:
        print("Drawing circle")

render(Circle())  # Type checks!
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 2.11 Magic Methods (Dunder Methods)

### Complete Reference

```python
class Vector:
    # Construction/Destruction
    def __new__(cls, *args):          # Create instance
        return super().__new__(cls)
    def __init__(self, x, y):          # Initialize instance
        self.x, self.y = x, y
    def __del__(self):                 # Cleanup
        pass
    
    # String representation
    def __repr__(self):                # Developer representation
        return f"Vector({self.x}, {self.y})"
    def __str__(self):                 # User representation
        return f"({self.x}, {self.y})"
    def __format__(self, spec):        # Custom formatting
        return f"Vector[{self.x:{spec}}, {self.y:{spec}}]"
    
    # Comparison
    def __eq__(self, other):           # ==
        return self.x == other.x and self.y == other.y
    def __lt__(self, other):           # <
        return (self.x, self.y) < (other.x, other.y)
    def __hash__(self):                # For use in sets/dicts
        return hash((self.x, self.y))
    
    # Arithmetic
    def __add__(self, other):          # +
        return Vector(self.x + other.x, self.y + other.y)
    def __mul__(self, scalar):         # *
        return Vector(self.x * scalar, self.y * scalar)
    def __rmul__(self, scalar):        # reverse *
        return self.__mul__(scalar)
    def __neg__(self):                 # unary -
        return Vector(-self.x, -self.y)
    def __abs__(self):                 # abs()
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    # Container behavior
    def __len__(self):                 # len()
        return 2
    def __getitem__(self, index):      # []
        return (self.x, self.y)[index]
    def __iter__(self):                # iteration
        yield self.x
        yield self.y
    def __contains__(self, value):     # in
        return value in (self.x, self.y)
    
    # Attribute access
    def __getattr__(self, name):       # Missing attribute
        raise AttributeError(f"No attribute: {name}")
    def __setattr__(self, name, value): # Set attribute
        super().__setattr__(name, value)
    
    # Callable
    def __call__(self, *args):         # instance()
        return self.x * args[0] + self.y * args[1]
    
    # Context manager
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

# Usage
v = Vector(3, 4)
print(repr(v))        # Vector(3, 4)
print(abs(v))         # 5.0
print(v + Vector(1,1)) # (4, 5)
print(3 * v)          # (9, 12)
print(v[0])           # 3
print(list(v))        # [3, 4]
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

# Part 3: Design Patterns

## 3.1 Creational Patterns

### Singleton

```java
// Java - Thread-safe Singleton
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

// Java - Enum Singleton (Best practice)
public enum SingletonEnum {
    INSTANCE;
    
    public void doSomething() {}
}
```

```python
# Python - Metaclass Singleton
class Singleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=Singleton):
    pass

# Python - Module-level (Simplest)
# mysingleton.py
class _Singleton:
    pass

instance = _Singleton()
```

### Factory Pattern

```java
// Java - Factory Method
public interface Vehicle {
    void drive();
}

public class Car implements Vehicle {
    public void drive() { System.out.println("Driving car"); }
}

public class Motorcycle implements Vehicle {
    public void drive() { System.out.println("Riding motorcycle"); }
}

public abstract class VehicleFactory {
    public abstract Vehicle createVehicle();
    
    public void deliverVehicle() {
        Vehicle v = createVehicle();
        v.drive();
    }
}

public class CarFactory extends VehicleFactory {
    public Vehicle createVehicle() { return new Car(); }
}
```

```python
# Python - Factory Method
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def drive(self): pass

class Car(Vehicle):
    def drive(self):
        print("Driving car")

class VehicleFactory(ABC):
    @abstractmethod
    def create_vehicle(self) -> Vehicle: pass
    
    def deliver_vehicle(self):
        vehicle = self.create_vehicle()
        vehicle.drive()

class CarFactory(VehicleFactory):
    def create_vehicle(self) -> Vehicle:
        return Car()
```

### Builder Pattern

```java
// Java - Builder
public class Computer {
    private String cpu;
    private int ram;
    private int storage;
    
    private Computer(Builder builder) {
        this.cpu = builder.cpu;
        this.ram = builder.ram;
        this.storage = builder.storage;
    }
    
    public static class Builder {
        private String cpu;
        private int ram;
        private int storage;
        
        public Builder cpu(String cpu) {
            this.cpu = cpu;
            return this;
        }
        
        public Builder ram(int ram) {
            this.ram = ram;
            return this;
        }
        
        public Builder storage(int storage) {
            this.storage = storage;
            return this;
        }
        
        public Computer build() {
            return new Computer(this);
        }
    }
}

// Usage
Computer pc = new Computer.Builder()
    .cpu("Intel i9")
    .ram(32)
    .storage(1000)
    .build();
```

```python
# Python - Builder
class Computer:
    def __init__(self):
        self.cpu = None
        self.ram = None
        self.storage = None

class ComputerBuilder:
    def __init__(self):
        self.computer = Computer()
    
    def cpu(self, cpu: str) -> 'ComputerBuilder':
        self.computer.cpu = cpu
        return self
    
    def ram(self, ram: int) -> 'ComputerBuilder':
        self.computer.ram = ram
        return self
    
    def storage(self, storage: int) -> 'ComputerBuilder':
        self.computer.storage = storage
        return self
    
    def build(self) -> Computer:
        return self.computer

# Usage
pc = (ComputerBuilder()
      .cpu("Intel i9")
      .ram(32)
      .storage(1000)
      .build())
```

### Abstract Factory

```java
// Java - Abstract Factory
public interface GUIFactory {
    Button createButton();
    Checkbox createCheckbox();
}

public class WindowsFactory implements GUIFactory {
    public Button createButton() { return new WindowsButton(); }
    public Checkbox createCheckbox() { return new WindowsCheckbox(); }
}

public class MacFactory implements GUIFactory {
    public Button createButton() { return new MacButton(); }
    public Checkbox createCheckbox() { return new MacCheckbox(); }
}
```

### Prototype Pattern

```java
// Java - Prototype
public abstract class Shape implements Cloneable {
    public abstract Shape clone();
}

public class Circle extends Shape {
    private int radius;
    
    public Circle(int radius) {
        this.radius = radius;
    }
    
    @Override
    public Shape clone() {
        return new Circle(this.radius);
    }
}
```

```python
# Python - Prototype
import copy

class Prototype:
    def clone(self):
        return copy.deepcopy(self)

class Circle(Prototype):
    def __init__(self, radius):
        self.radius = radius
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 3.2 Structural Patterns

### Adapter Pattern

```java
// Java - Adapter
public interface MediaPlayer {
    void play(String filename);
}

public class VLCPlayer {
    public void playVLC(String filename) {
        System.out.println("Playing VLC: " + filename);
    }
}

// Adapter
public class VLCAdapter implements MediaPlayer {
    private VLCPlayer vlcPlayer;
    
    public VLCAdapter() {
        vlcPlayer = new VLCPlayer();
    }
    
    @Override
    public void play(String filename) {
        vlcPlayer.playVLC(filename);
    }
}
```

```python
# Python - Adapter
class VLCPlayer:
    def play_vlc(self, filename):
        print(f"Playing VLC: {filename}")

class MediaPlayer:
    def play(self, filename): pass

class VLCAdapter(MediaPlayer):
    def __init__(self):
        self.vlc = VLCPlayer()
    
    def play(self, filename):
        self.vlc.play_vlc(filename)
```

### Decorator Pattern

```java
// Java - Decorator
public interface Coffee {
    double getCost();
    String getDescription();
}

public class SimpleCoffee implements Coffee {
    public double getCost() { return 1.0; }
    public String getDescription() { return "Simple coffee"; }
}

public abstract class CoffeeDecorator implements Coffee {
    protected Coffee coffee;
    
    public CoffeeDecorator(Coffee coffee) {
        this.coffee = coffee;
    }
}

public class MilkDecorator extends CoffeeDecorator {
    public MilkDecorator(Coffee coffee) {
        super(coffee);
    }
    
    public double getCost() {
        return coffee.getCost() + 0.5;
    }
    
    public String getDescription() {
        return coffee.getDescription() + ", milk";
    }
}

// Usage
Coffee coffee = new MilkDecorator(new SimpleCoffee());
// Cost: 1.5, Description: "Simple coffee, milk"
```

```python
# Python - Decorator (using actual decorators!)
class Coffee:
    def get_cost(self): return 1.0
    def get_description(self): return "Simple coffee"

def milk(cls):
    original_cost = cls.get_cost
    original_desc = cls.get_description
    
    def new_cost(self):
        return original_cost(self) + 0.5
    
    def new_desc(self):
        return original_desc(self) + ", milk"
    
    cls.get_cost = new_cost
    cls.get_description = new_desc
    return cls

@milk
class MilkCoffee(Coffee):
    pass
```

### Facade Pattern

```java
// Java - Facade
public class ComputerFacade {
    private CPU cpu;
    private Memory memory;
    private HardDrive hardDrive;
    
    public ComputerFacade() {
        cpu = new CPU();
        memory = new Memory();
        hardDrive = new HardDrive();
    }
    
    public void start() {
        cpu.freeze();
        memory.load(hardDrive.read());
        cpu.execute();
    }
}
```

### Proxy Pattern

```java
// Java - Proxy
public interface Image {
    void display();
}

public class RealImage implements Image {
    private String filename;
    
    public RealImage(String filename) {
        this.filename = filename;
        loadFromDisk();
    }
    
    private void loadFromDisk() {
        System.out.println("Loading: " + filename);
    }
    
    public void display() {
        System.out.println("Displaying: " + filename);
    }
}

// Proxy - Lazy loading
public class ImageProxy implements Image {
    private String filename;
    private RealImage realImage;
    
    public ImageProxy(String filename) {
        this.filename = filename;
    }
    
    public void display() {
        if (realImage == null) {
            realImage = new RealImage(filename);
        }
        realImage.display();
    }
}
```

### Composite Pattern

```java
// Java - Composite
public interface Component {
    void operation();
}

public class Leaf implements Component {
    public void operation() {
        System.out.println("Leaf operation");
    }
}

public class Composite implements Component {
    private List<Component> children = new ArrayList<>();
    
    public void add(Component c) { children.add(c); }
    public void remove(Component c) { children.remove(c); }
    
    public void operation() {
        for (Component child : children) {
            child.operation();
        }
    }
}
```

### Bridge Pattern

```java
// Java - Bridge
// Separates abstraction from implementation
public interface Device {
    void turnOn();
    void turnOff();
}

public class TV implements Device {
    public void turnOn() { System.out.println("TV on"); }
    public void turnOff() { System.out.println("TV off"); }
}

public abstract class Remote {
    protected Device device;
    
    public Remote(Device device) {
        this.device = device;
    }
    
    public abstract void togglePower();
}

public class BasicRemote extends Remote {
    private boolean on = false;
    
    public BasicRemote(Device device) {
        super(device);
    }
    
    public void togglePower() {
        if (on) {
            device.turnOff();
        } else {
            device.turnOn();
        }
        on = !on;
    }
}
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 3.3 Behavioral Patterns

### Strategy Pattern

```java
// Java - Strategy
public interface PaymentStrategy {
    void pay(int amount);
}

public class CreditCardPayment implements PaymentStrategy {
    private String cardNumber;
    
    public CreditCardPayment(String cardNumber) {
        this.cardNumber = cardNumber;
    }
    
    public void pay(int amount) {
        System.out.println("Paid " + amount + " via credit card");
    }
}

public class PayPalPayment implements PaymentStrategy {
    public void pay(int amount) {
        System.out.println("Paid " + amount + " via PayPal");
    }
}

public class ShoppingCart {
    private PaymentStrategy paymentStrategy;
    
    public void setPaymentStrategy(PaymentStrategy strategy) {
        this.paymentStrategy = strategy;
    }
    
    public void checkout(int amount) {
        paymentStrategy.pay(amount);
    }
}
```

```python
# Python - Strategy (using functions)
from typing import Callable

def credit_card_payment(amount: int):
    print(f"Paid {amount} via credit card")

def paypal_payment(amount: int):
    print(f"Paid {amount} via PayPal")

class ShoppingCart:
    def __init__(self, payment_strategy: Callable[[int], None]):
        self.payment_strategy = payment_strategy
    
    def checkout(self, amount: int):
        self.payment_strategy(amount)

# Usage
cart = ShoppingCart(credit_card_payment)
cart.checkout(100)
```

### Observer Pattern

```java
// Java - Observer
public interface Observer {
    void update(String message);
}

public class Subject {
    private List<Observer> observers = new ArrayList<>();
    private String state;
    
    public void attach(Observer observer) {
        observers.add(observer);
    }
    
    public void detach(Observer observer) {
        observers.remove(observer);
    }
    
    public void setState(String state) {
        this.state = state;
        notifyObservers();
    }
    
    private void notifyObservers() {
        for (Observer observer : observers) {
            observer.update(state);
        }
    }
}
```

```python
# Python - Observer
from abc import ABC, abstractmethod
from typing import List

class Observer(ABC):
    @abstractmethod
    def update(self, message: str): pass

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []
        self._state = None
    
    def attach(self, observer: Observer):
        self._observers.append(observer)
    
    def detach(self, observer: Observer):
        self._observers.remove(observer)
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        self._state = value
        self._notify()
    
    def _notify(self):
        for observer in self._observers:
            observer.update(self._state)
```

### Command Pattern

```java
// Java - Command
public interface Command {
    void execute();
    void undo();
}

public class LightOnCommand implements Command {
    private Light light;
    
    public LightOnCommand(Light light) {
        this.light = light;
    }
    
    public void execute() {
        light.turnOn();
    }
    
    public void undo() {
        light.turnOff();
    }
}

public class RemoteControl {
    private Stack<Command> history = new Stack<>();
    
    public void executeCommand(Command cmd) {
        cmd.execute();
        history.push(cmd);
    }
    
    public void undo() {
        if (!history.isEmpty()) {
            history.pop().undo();
        }
    }
}
```

### Template Method Pattern

```java
// Java - Template Method
public abstract class DataProcessor {
    // Template method
    public final void process() {
        readData();
        processData();
        writeData();
    }
    
    protected abstract void readData();
    protected abstract void processData();
    
    // Default implementation
    protected void writeData() {
        System.out.println("Writing to file...");
    }
}

public class CSVProcessor extends DataProcessor {
    protected void readData() {
        System.out.println("Reading CSV...");
    }
    
    protected void processData() {
        System.out.println("Processing CSV data...");
    }
}
```

### State Pattern

```java
// Java - State
public interface State {
    void handle(Context context);
}

public class ConcreteStateA implements State {
    public void handle(Context context) {
        System.out.println("State A handling...");
        context.setState(new ConcreteStateB());
    }
}

public class ConcreteStateB implements State {
    public void handle(Context context) {
        System.out.println("State B handling...");
        context.setState(new ConcreteStateA());
    }
}

public class Context {
    private State state;
    
    public Context(State state) {
        this.state = state;
    }
    
    public void setState(State state) {
        this.state = state;
    }
    
    public void request() {
        state.handle(this);
    }
}
```

### Chain of Responsibility

```java
// Java - Chain of Responsibility
public abstract class Handler {
    protected Handler next;
    
    public Handler setNext(Handler next) {
        this.next = next;
        return next;
    }
    
    public abstract void handle(Request request);
}

public class AuthHandler extends Handler {
    public void handle(Request request) {
        if (request.isAuthenticated()) {
            if (next != null) next.handle(request);
        } else {
            System.out.println("Authentication failed");
        }
    }
}

public class ValidationHandler extends Handler {
    public void handle(Request request) {
        if (request.isValid()) {
            if (next != null) next.handle(request);
        } else {
            System.out.println("Validation failed");
        }
    }
}

// Usage
Handler chain = new AuthHandler();
chain.setNext(new ValidationHandler())
     .setNext(new ProcessingHandler());
chain.handle(request);
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 3.4 Concurrency Patterns

### Producer-Consumer

```java
// Java - Producer-Consumer with BlockingQueue
public class ProducerConsumer {
    private final BlockingQueue<Integer> queue = new LinkedBlockingQueue<>(10);
    
    public void produce() throws InterruptedException {
        int value = 0;
        while (true) {
            queue.put(value++);
            System.out.println("Produced: " + value);
        }
    }
    
    public void consume() throws InterruptedException {
        while (true) {
            int value = queue.take();
            System.out.println("Consumed: " + value);
        }
    }
}
```

```python
# Python - Producer-Consumer with asyncio
import asyncio
from asyncio import Queue

async def producer(queue: Queue):
    for i in range(10):
        await queue.put(i)
        print(f"Produced: {i}")
        await asyncio.sleep(0.1)

async def consumer(queue: Queue):
    while True:
        item = await queue.get()
        print(f"Consumed: {item}")
        queue.task_done()

async def main():
    queue = Queue()
    producers = [asyncio.create_task(producer(queue))]
    consumers = [asyncio.create_task(consumer(queue)) for _ in range(3)]
    await asyncio.gather(*producers)
    await queue.join()
```

### Thread Pool

```java
// Java - Custom Thread Pool
ExecutorService executor = new ThreadPoolExecutor(
    4,                      // Core pool size
    10,                     // Max pool size
    60L, TimeUnit.SECONDS,  // Keep-alive time
    new LinkedBlockingQueue<>(100),  // Work queue
    new ThreadPoolExecutor.CallerRunsPolicy()  // Rejection policy
);
```

### Read-Write Lock

```java
// Java - Read-Write Lock
public class Cache<K, V> {
    private final Map<K, V> map = new HashMap<>();
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    
    public V get(K key) {
        lock.readLock().lock();
        try {
            return map.get(key);
        } finally {
            lock.readLock().unlock();
        }
    }
    
    public void put(K key, V value) {
        lock.writeLock().lock();
        try {
            map.put(key, value);
        } finally {
            lock.writeLock().unlock();
        }
    }
}
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 3.5 Architectural Patterns

### MVC (Model-View-Controller)

```
┌─────────┐     updates     ┌─────────┐
│  View   │◀────────────────│  Model  │
└────┬────┘                 └────▲────┘
     │                           │
     │ user action               │ updates
     ▼                           │
┌─────────────┐                  │
│ Controller  │──────────────────┘
└─────────────┘
```

### Repository Pattern

```java
// Java - Repository Pattern
public interface UserRepository {
    User findById(Long id);
    List<User> findAll();
    User save(User user);
    void delete(User user);
}

public class JpaUserRepository implements UserRepository {
    private final EntityManager em;
    
    public User findById(Long id) {
        return em.find(User.class, id);
    }
    // ... other implementations
}

// Service layer uses repository
public class UserService {
    private final UserRepository userRepository;
    
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
    
    public User getUser(Long id) {
        return userRepository.findById(id);
    }
}
```

### Dependency Injection

```java
// Java - Constructor Injection (Preferred)
public class OrderService {
    private final OrderRepository repository;
    private final NotificationService notifications;
    
    // Dependencies injected via constructor
    public OrderService(OrderRepository repository, 
                       NotificationService notifications) {
        this.repository = repository;
        this.notifications = notifications;
    }
}

// With Spring
@Service
public class OrderService {
    private final OrderRepository repository;
    
    @Autowired
    public OrderService(OrderRepository repository) {
        this.repository = repository;
    }
}
```

```python
# Python - Dependency Injection
class OrderService:
    def __init__(self, repository: OrderRepository, 
                 notifications: NotificationService):
        self.repository = repository
        self.notifications = notifications

# Using a simple DI container
class Container:
    _instances = {}
    _factories = {}
    
    @classmethod
    def register(cls, interface, factory):
        cls._factories[interface] = factory
    
    @classmethod
    def resolve(cls, interface):
        if interface not in cls._instances:
            cls._instances[interface] = cls._factories[interface]()
        return cls._instances[interface]
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

# Part 4: FAANG-Specific Topics

## 4.1 System Design Fundamentals

### Key Concepts to Master

| Concept | What to Know |
|---------|--------------|
| Load Balancing | Round Robin, Least Connections, Consistent Hashing |
| Caching | Redis, Memcached, Cache invalidation strategies |
| Database Sharding | Horizontal partitioning, Shard keys |
| Message Queues | Kafka, RabbitMQ, SQS |
| CDN | Content delivery, Edge caching |
| Microservices | Service discovery, API Gateway |
| CAP Theorem | Consistency, Availability, Partition tolerance |
| ACID vs BASE | Strong vs eventual consistency |

### System Design Template

```
1. Requirements Clarification (5 min)
   - Functional requirements
   - Non-functional requirements (scale, latency, availability)

2. Back-of-envelope Estimation (5 min)
   - Traffic estimates (QPS)
   - Storage estimates
   - Bandwidth estimates

3. High-Level Design (10 min)
   - Major components
   - Data flow

4. Detailed Design (15 min)
   - Database schema
   - API design
   - Key algorithms

5. Bottlenecks & Trade-offs (5 min)
   - Single points of failure
   - Scalability improvements
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 4.2 Distributed Systems Concepts

### Key Topics

- **Consensus Algorithms**: Paxos, Raft
- **Distributed Transactions**: 2PC, Saga pattern
- **Replication**: Master-slave, Master-master
- **Partitioning**: Range, Hash, Consistent hashing
- **Failure Detection**: Heartbeats, Gossip protocol
- **Clock Synchronization**: Logical clocks, Vector clocks

### Interview Topics

- [ ] How does consistent hashing work?
- [ ] Explain the CAP theorem with examples
- [ ] How do you handle distributed transactions?
- [ ] What is eventual consistency?
- [ ] How does a distributed cache work?

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 4.3 Code Quality & Best Practices

### SOLID Principles

| Principle | Description |
|-----------|-------------|
| **S**ingle Responsibility | A class should have one reason to change |
| **O**pen/Closed | Open for extension, closed for modification |
| **L**iskov Substitution | Subtypes must be substitutable for base types |
| **I**nterface Segregation | Many specific interfaces > one general interface |
| **D**ependency Inversion | Depend on abstractions, not concretions |

### Clean Code Practices

```java
// ❌ Bad
public void p(List<int[]> l) {
    for (int[] x : l) {
        if (x[0] > 0) {
            // ...
        }
    }
}

// ✅ Good
public void processTransactions(List<Transaction> transactions) {
    for (Transaction transaction : transactions) {
        if (transaction.isValid()) {
            processValidTransaction(transaction);
        }
    }
}
```

[⬆ Back to Table of Contents](#-table-of-contents)

---

# Part 5: Resources

## 5.1 YouTube Channels & Playlists

### Java Advanced

| Channel | Topic | Link |
|---------|-------|------|
| **Defog Tech** | JVM Internals, GC | Search: "Defog Tech Java" |
| **Java Brains** | Spring, Microservices | Search: "Java Brains" |
| **Baeldung** | Advanced Java | Search: "Baeldung Java" |
| **Tech Dummies** | System Design, Java | Search: "Tech Dummies" |

### Python Advanced

| Channel | Topic | Link |
|---------|-------|------|
| **mCoding** | Python Internals | Search: "mCoding Python" |
| **ArjanCodes** | Design Patterns | Search: "ArjanCodes" |
| **Corey Schafer** | Advanced Python | Search: "Corey Schafer Python" |
| **Tech With Tim** | Python Projects | Search: "Tech With Tim" |

### System Design

| Channel | Topic | Link |
|---------|-------|------|
| **System Design Interview** | Mock Interviews | Search: "System Design Interview" |
| **Gaurav Sen** | System Design | Search: "Gaurav Sen" |
| **Exponent** | FAANG Interviews | Search: "Exponent" |
| **ByteByteGo** | System Design | Search: "ByteByteGo" |

### Design Patterns

| Channel | Topic | Link |
|---------|-------|------|
| **Christopher Okhravi** | GoF Patterns | Search: "Christopher Okhravi" |
| **Derek Banas** | Design Patterns | Search: "Derek Banas Design Patterns" |
| **Fireship** | Quick Overviews | Search: "Fireship Design Patterns" |

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 5.2 Books

### Java

| Book | Author | Focus |
|------|--------|-------|
| Effective Java (3rd Ed) | Joshua Bloch | Best practices |
| Java Concurrency in Practice | Brian Goetz | Concurrency |
| Modern Java in Action | Raoul-Gabriel Urma | Streams, Lambdas |
| Java Performance | Scott Oaks | JVM Tuning |

### Python

| Book | Author | Focus |
|------|--------|-------|
| Fluent Python | Luciano Ramalho | Pythonic code |
| Python Cookbook | David Beazley | Advanced recipes |
| High Performance Python | Gorelick & Ozsvald | Optimization |

### Design Patterns & Architecture

| Book | Author | Focus |
|------|--------|-------|
| Design Patterns (GoF) | Gang of Four | Classic patterns |
| Head First Design Patterns | Freeman & Robson | Visual learning |
| Clean Architecture | Robert Martin | Architecture |
| Designing Data-Intensive Applications | Martin Kleppmann | Distributed Systems |

[⬆ Back to Table of Contents](#-table-of-contents)

---

## 5.3 Practice Platforms

| Platform | Best For | Link |
|----------|----------|------|
| LeetCode | DSA, Patterns | leetcode.com |
| HackerRank | Language Skills | hackerrank.com |
| System Design Primer | System Design | github.com/donnemartin/system-design-primer |
| Educative.io | Interactive Learning | educative.io |
| Pramp | Mock Interviews | pramp.com |
| Interviewing.io | Practice with Engineers | interviewing.io |

---

## 📅 Study Plan Suggestion

### Week 1-2: Java Deep Dive
- [ ] JVM Internals
- [ ] Memory Management & GC
- [ ] Concurrency

### Week 3-4: Python Deep Dive
- [ ] Decorators, Generators
- [ ] Metaclasses, Descriptors
- [ ] asyncio

### Week 5-6: Design Patterns
- [ ] Creational Patterns
- [ ] Structural Patterns
- [ ] Behavioral Patterns

### Week 7-8: System Design
- [ ] Fundamentals
- [ ] Common Systems (URL Shortener, Twitter, etc.)
- [ ] Mock Interviews

---

> **Remember**: Consistency beats intensity. Study a little every day rather than cramming.

> **Pro Tip**: Implement each design pattern yourself in both Java AND Python. The muscle memory will help in interviews!

---

*Good luck on your journey to becoming a God-Tier Software Engineer! 🚀*

[⬆ Back to Table of Contents](#-table-of-contents)