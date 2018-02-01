# Why Python is Slow: Looking Under the Hood
Original link [https://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/](https://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/)

**Cython and Numba** [http://jakevdp.github.io/blog/2013/06/15/numba-vs-cython-take-2/](http://jakevdp.github.io/blog/2013/06/15/numba-vs-cython-take-2/)

We've all heard it before: Python is slow.

When I teach courses on Python for scientific computing, I make this point very early in the course, and tell the students why: it boils down to Python being a dynamically typed, interpreted language, where values are stored not in dense buffers but in scattered objects. And then I talk about how to get around this by using NumPy, SciPy, and related tools for vectorization of operations and calling into compiled code, and go on from there.

But I realized something recently: despite the relative accuracy of the above statements, the words "dynamically-typed-interpreted-buffers-vectorization-compiled" probably mean very little to somebody attending an intro programming seminar. The jargon does little to enlighten people about what's actually going on "under the hood", so to speak.

So I decided I would write this post, and dive into the details that I usually gloss over. Along the way, we'll take a look at using Python's standard library to introspect the goings-on of CPython itself. So whether you're a novice or experienced programmer, I hope you'll learn something from the following exploration.

# Why Python is Slow
Python is slower than Fortran and C for a variety of reasons:


## 1. Python is Dynamically Typed rather than Statically Typed.

What this means is that at the time the program executes, the interpreter doesn't know the type of the variables that are defined. The difference between a C variable (I'm using C as a stand-in for compiled languages) and a Python variable is summarized by this diagram:

![](http://jakevdp.github.io/images/cint_vs_pyint.png)

So if you write the following in C:

```C
/* C code */
int a = 1;
int b = 2;
int c = a + b;
```

the C compiler knows from the start that a and b are integers: they simply can't be anything else! With this knowledge, it can call the routine which adds two integers, returning another integer which is just a simple value in memory. As a rough schematic, the sequence of events looks like this:

### C Addition

1. Assign `<int> 1` to `a`
2. Assign `<int> 2` to `b`
call `binary_add<int`, `int>(a, b)`
3. Assign the result to `c`

The equivalent code in Python looks like this:

```python
a = 1
b = 2
c = a + b
```
here the interpreter knows only that 1 and 2 are objects, but not what type of object they are. So the The interpreter must inspect PyObject_HEAD for each variable to find the type information, and then call the appropriate summation routine for the two types. Finally it must create and initialize a new Python object to hold the return value. The sequence of events looks roughly like this:

### Python Addition

1. Assign 1 to a

- 1a. Set `a->PyObject_HEAD->typecode` to integer
- 1b. Set `a->val = 1`

2. Assign 2 to b

- 2a. Set `b->PyObject_HEAD->typecode` to integer
- 2b. Set `b->val = 2`
 
3. call binary_add(a, b)

- 3a. find typecode in `a->PyObject_HEAD`
- 3b. `a` is an integer; value is `a->val`
- 3c. find typecode in `b->PyObject_HEAD`
- 3d. `b` is an integer; value is `b->val`
- 3e. call `binary_add<int`, `int>(a->val, b->val)`
- 3f. result of this is result, and is an integer.

4. Create a Python object c

4a. set `c->PyObject_HEAD->typecode` to integer
4b. set `c->val` to result

The dynamic typing means that there are a lot more steps involved with any operation. This is a primary reason that Python is slow compared to C for operations on numerical data.

## 2. Python is interpreted rather than compiled.

We saw above one difference between interpreted and compiled code. A smart compiler can look ahead and optimize for repeated or unneeded operations, which can result in speed-ups. Compiler optimization is its own beast, and I'm personally not qualified to say much about it, so I'll stop there. For some examples of this in action, you can take a look at my previous post on Numba and Cython.

## 3. Python's Object Model can lead to inefficient Memory Access

We saw above the extra type info layer when moving from a C integer to a Python integer. Now imagine you have many such integers and want to do some sort of batch operation on them. In Python you might use the standard List object, while in C you would likely use some sort of buffer-based array.

A NumPy array in its simplest form is a Python object build around a C array. That is, it has a pointer to a contiguous data buffer of values. A Python list, on the other hand, has a pointer to a contiguous buffer of pointers, each of which points to a Python object which in turn has references to its data (in this case, integers). This is a schematic of what the two might look like:

![http://jakevdp.github.io/images/array_vs_list.png](http://jakevdp.github.io/images/array_vs_list.png)

It's easy to see that if you're doing some operation which steps through data in sequence, the numpy layout will be much more efficient than the Python layout, both in the cost of storage and the cost of access.



## So Why Use Python?
Given this inherent inefficiency, why would we even think about using Python? Well, it comes down to this: Dynamic typing makes Python easier to use than C. It's extremely flexible and forgiving, this flexibility leads to efficient use of development time, and on those occasions that you really need the optimization of C or Fortran, Python offers easy hooks into compiled libraries. It's why Python use within many scientific communities has been continually growing. With all that put together, Python ends up being an extremely efficient language for the overall task of doing science with code.

...

# Conclusion
Python is slow. And one big reason for that, as we've seen, is the type indirection under the hood which makes Python quick, easy, and fun for the developer. And as we've seen, Python itself offers tools that can be used to hack into the Python objects themselves.

I hope that this was made more clear through this exploration of the differences between various objects, and some liberal mucking around in the internals of CPython itself. This exercise was extremely enlightening for me, and I hope it was for you as well... Happy hacking!