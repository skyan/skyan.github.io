+++
title = 'Google abseil开源项目介绍'
date = 2023-06-01T14:59:19+08:00
author = "Skyan"
tags = ["programming", "open source"]
ShowToc = true
ShowBreadCrumbs = true
+++

[Google abseil](https://abseil.io/)是Google开源的优秀C++基础库，持续维护并且持续迭代。该库的代码质量和工程化水平属于业界顶级，值得我们在实际生产中使用和学习。不仅要善于使用abseil库，还要多看abseil的文档和代码，从中学习Google业界领先的C++经验。

这里先介绍几个abseil库的经典组件：

## 容器
> Recommendation
> Prefer absl::flat_hash_map or absl::flat_hash_set in most new code (see above).
> Use absl::node_hash_map or absl::node_hash_set when pointer stability of both keys and values is required (rare), or for code migrations from other containers with this property. Note: Do not use popularity as a guide. You will see the “node” containers used a lot, but only because it was safe to migrate code to them from other containers.

大部分情况下，直接用absl::flat_hash_map/set 代替std::unorderd_map/set，如果需要保持key value指针的稳定性，才考虑使用absl::node_hash_map/set (这个比较少见)
理由：1. 查询性能；2. 省内存

## Random
非常好用的随机数生成器类，可以随机抽样使用，我们经常在抽样流量的场景使用：
```cpp
 #include "absl/random/random.h"
{
    thread_local absl::BitGen bitgen; // seed in thread local
    if (absl::Uniform(bitgen, 0, 100) >= sample_ratio){
        return true;
    }
}
```

## String Format

printf平替，Go的fmt包的对标版本，比如这样用：
```cpp
#include "absl/strings/str_format.h"

std::string s = absl::StrFormat("Welcome to %s, Number %d!", "The Village", 6);
EXPECT_EQ("Welcome to The Village, Number 6!", s);
```

## Strings

各种字符串操作，极其推荐，不用重复造轮子
比如高效的string_view，可以看这篇介绍。

比如切割字符串：

```cpp
// Splits the given string on commas. Returns the results in a
// vector of strings. (Data is copied once.)
std::vector<std::string> v = absl::StrSplit("a,b,c", ',');  // Can also use ","
// v[0] == "a", v[1] == "b", v[2] == "c"

// Splits the string as in the previous example, except that the results
// are returned as `absl::string_view` objects, avoiding copies. Note that
// because we are storing the results within `absl::string_view` objects, we
// have to ensure that the input string outlives any results.
std::vector<absl::string_view> v = absl::StrSplit("a,b,c", ',');
// v[0] == "a", v[1] == "b", v[2] == "c"
```

比如高效地拼接字符串：
```cpp
// Efficient code
std::string s1 = "A string";
std::string another = " and another string";
absl::StrAppend(&s1, " and some other string", another);

// absl::StrCat() can merge an arbitrary number of strings
std::string s1;
s1 = absl::StrCat("A string ", " another string", "yet another string");

// StrCat() also can mix types, including std::string, string_view, literals,
// and more.
std::string s1;
std::string s2 = "Foo";
absl::string_view sv1 = MyFunction();
s1 = absl::StrCat(s2, sv1, "a literal");
```

比如join字符串：
```cpp
std::vector<std::string> v = {"foo", "bar", "baz"};
std::string s = absl::StrJoin(v, "-");
// Produces the string "foo-bar-baz"
```

比如字符串模板：
```cpp
// Best. Using absl::Substitute() is easier to read and to understand.
std::string GetErrorMessage(absl::string_view op, absl::string_view user, int id) {
  return absl::Substitute("Error in $0 for user $1 ($2)", op, user, id);
}
```

比如字符串匹配：
```cpp
// Assume "msg" is a line from a logs entry
if (absl::StrContains(msg, "ERROR")) {
  *has_error = true;
}
if (absl::StrContains(msg, "WARNING")) {
  *has_warning = true;
}
```

比如文本转数值：
* **absl::SimpleAtoi()**  converts a string into integral types.
* **absl::SimpleAtof()** converts a string into a float.
* **absl::SimpleAtod()** converts a string into a double.
* **absl::SimpleAtob()** converts a string into a boolean.

## Status

写过Go代码的同学一定不陌生，这个Status类对标就是Go标准库里的Error（谁先谁后有待考证），所以如果想返回一个状态，同时附带一些错误信息，用Status类再合适不过了。

一般经常用Status作为函数的返回值，取代int这样的返回类型，返回状态更加丰富，例如：
```cpp
absl::Status MyFunction(absl::string_view filename, ...) {
  ...
  // encounter error
  if (error condition) {
    return absl::InvalidArgumentError("bad mode");
  }
  // else, return OK
  return absl::OkStatus();
}
```

## Synchronization

同步库里有大量多线程同步的实用类，可以各取所需：

* mutex.hProvides primitives for managing locks on resources. A mutex is the most important primitive in this library and the building block for most all concurrency utilities.
* notification.hProvides a simple mechanism for notifying threads of events.
* barrier.h and blocking_counter.hProvides synchronization abstractions for cumulative events.
* base/thread_annotations.hProvides macros for documenting the locking policy of multi-threaded code, and providing warnings and errors for misuse of such locks.
* base/call_once.h Provides an Abseil version of std::call_once() for invoking a callable object exactly once across all threads.

## Time
写过Go代码的同学同样不陌生，这些类对标的就是Go的time标准库。Duration，Now这些名字熟悉的不能再熟悉了。

基本上可以取代gettimeofday这样的函数调用，用来做高精度计时函数也很好用。

## C++ Tips
abseil项目的另外一个巨大的贡献就是写了一系列C++的Tips文档，把C++17以后的C++新的标准，新的变化，新的工业级实用技巧都整理出一个个小Tips文档，每一篇都值得希望写好C++代码的同学学习研究。

让我印象比较深刻的几篇：
1. Tip of the Week #1: [string_view](https://abseil.io/tips/1)      一定要了解什么是string_view以及如何用好它
2. Tip of the Week #11: [Return Policy](https://abseil.io/tips/11)   一定要看下RVO
3. Tip of the Week #49: [Argument-Dependent Lookup](https://abseil.io/tips/49)    值得精读
4. Tip of the Week #64: [Raw String Literals](https://abseil.io/tips/64) 
5. Tip of the Week #76: [Use absl::Status](https://abseil.io/tips/76) 
6. Tip of the Week #140: [Constants: Safe Idioms](https://abseil.io/tips/140)  

还有很多非常干货的文章，对C++工业级开发感兴趣的同学都得通读一遍。

## Fast Tips
今年开始，abseil项目关于性能也有一系列文章，对C++高性能开发感兴趣的也值得一读

## 结论
abseil项目值得学习和在生产环境应用，是掌握现代C++开发的必备学习材料。
