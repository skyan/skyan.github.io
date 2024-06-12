+++
title = '并行编程之内存管理(一)'
date = 2022-06-06T16:15:19+08:00
author = "Skyan"
tags = ["programming", "parallel programming"]
ShowToc = false
ShowBreadCrumbs = false
+++

## 前言
随着现代处理器的发展和多核多CPU体系结构的大面积应用，C和C++编程面临着更加复杂和陡峭的学习曲线。特别是基于多线程带来的并行编程，带来了很多内存并行访问的问题。这需要非常专业的知识，深入了解CPU指令集，内存访问，CPU Cache等体系结构的底层知识，才能正确写好高性能和安全的并行程序。最近十多年，学术和工业界在并行编程方面进行了非常多创新的探索和研究，总结出一套优秀的编程实践和并行内存管理组件，并在Linux内核和大型开源软件中广泛应用。这里选取和内存对象管理有关的三个编程组件进行介绍，分别是引用计数Reference Counting，Hazard Pointer和RCU，都属于延迟处理类型的组件。本文目的一是为了个人学习的总结，另外也是给更多感兴趣的同学以启发。

## 引用计数
引用计数的思想很简单，通过原子变量来追踪对象的引用数量，来防止错误地销毁对象。这种思想最早可以追溯到20世纪40年代：当时工人们如果要修理危险的大型机械，他们会在进入机器之前，在机器开关上面挂一把锁，防止他在里面的时候被其他人误开机器。这也说明了引用计数的作用：通过计数来管理对象的生命周期。

以shared pointer为例，参考gcc shared_ptr实现，做了一些简化，样例代码如下：
```cpp
// 计数管理类
template <typename T>
class RefCount {
 public:
  RefCount(T *p = nullptr) : ptr_(p), cnt_(1) {}
  ~RefCount() {
    // 如果计数为0，销毁自己
    if (Release()) {
      delete this;
    }
  }
  RefCount(const RefCount<T> &rc) = delete;
  void operator=(const RefCount<T> &rc) = delete;
  // 原子++
  void AddRef() {
    ++cnt_;
  }
  // 原子--，如果计数为0，销毁保存的对象
  bool Release() {
    if (cnt_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete ptr_;
      return true;
    }
    return false;
  }
  // 返回管理的指针
  T *Get() const noexcept { return ptr_; }

 private:
  T *ptr_;  // 管理的对象指针
  std::atomic_int32_t cnt_;   // 原子计数
};

// 包装为智能指针
template <typename T>
class SharedPtr {
 public:
  SharedPtr() noexcept : rc_(nullptr) {}
  SharedPtr(T *p) : rc_(nullptr) { rc_ = new RefCount(p); }  // 创建引用计数对象

  SharedPtr(const SharedPtr<T> &sp) : rc_(sp.rc_) {
    if (rc_ != nullptr) {
      rc_->AddRef();     // 增加计数
    }
  }
  // 计数减一
  ~SharedPtr() { rc_->Release(); }
  // 获取裸指针
  T *Get() { return rc_->Get(); }
  // 拷贝操作，当前指针计数减一，被拷贝指针计数加一
  SharedPtr &operator=(const SharedPtr<T> &p) {
    RefCount<T> *tmp = p.rc_;
    if (tmp != rc_) {
      if (tmp != nullptr) {
        tmp->AddRef();    // 拷贝调用也增加计数
      }
      if (rc_ != nullptr) {
        rc_->Release();   // 当前管理的计数减少引用
      }
      rc_ = tmp;
    }
    return *this;
  }

 private:
  RefCount<T> *rc_;  // 计数对象
};
```

整个代码是比较易懂的。一般来说引用计数都是采用原子变量在构造和析构的时候分别+1和-1的。当计数为0的时候，则销毁管理的对象。

但需要注意的是，cpp标准库的shared_ptr以及上面的样例代码都不是线程安全的。如果两个线程同时操作一个SharedPtr对象，那么很可能会导致内存错误。典型的问题就在SharedPtr的Get方法以及拷贝构造函数里，可以明细看出拷贝构造函数并不是线程安全的实现，并且Get方法也很有可能获取一个已经释放的对象指针。这也是生产代码容易误用shared pointer之处。

为了让引用计数更加鲁棒，还需要进一步升级。以FB的folly库的AtomicSharedPtr为代表，实现了原子变更“对象指针+引用计数+alias对象”的功能，真正实现线程安全的原子引用计数对象管理。

如果要实现一个AtomicSharedPtr，需要解决的一个问题就是如何用一个原子操作同时变更指针+引用计数。好在x64平台的虚拟内存地址有个机制是地址的高16位都是0，可以利用这16位做引用计数，就可以基于64位的CAS实现同时变更指针+引用计数的功能了。这也是folly中PackedSyncPtr基本原理。基于这个功能，就可以实现一个线程安全的AtomicSharedPtr，用来在多线程环境下管理对象的生命周期了。详情可以参考folly代码。

引用计数的优点在于不会死锁(lock-free)，自动回收内存，不感知线程，和TLS无关。缺点是高并发读的时候竞争会比较激烈，高并发读写性能并不太好。在TLS支持缺失，以及需要避免死锁，或者需要自动回收内存的场景下，适合用引用计数的方法。
