---
layout: post
tags: programming
author: Chunpai
---

This post is about how to write a Python wrapper for C/C++ program using "Simple Wrapper Interface Generator" (SWIG), which turns C code into a Python module and enables calling C functions in python directly and speed up the running time of code.

* TOC
{: toc}
## SWIG

> SWIG is an interface compiler that connects programs written in C and C++ with scripting languages such as Perl, Python, Ruby, and Tcl. It works by taking the declarations found in C/C++ header files and using them to generate the wrapper code that scripting languages need to access the underlying C/C++ code. In addition, SWIG provides a variety of customization features that let you tailor the wrapping process to suit your application.



SWIG requires an interface file fed into the generator and it will automatically generate two files: `*_wrap.c` and `*.py`, where `*_wrap.c` is used to generate the file `*.so` file and `*.py` provided the functionality of Python interface for C code.

 

## A Quick Example

For example, we have C code called `example.c` , which contains 1 global variable and 3 functions.

```c
#include <time.h>

double My_variable = 3.0;

int fact(int n) {
  if (n <= 1) return 1;
  else return n*fact(n-1);
}

int my_mod(int x, int y) {
  return (x%y);
}

char *get_time()
{
  time_t ltime;
  time(&ltime);
  return ctime(&ltime);
}
```

Now, we would like to call those functions, such as `fact()` in our Python code `test.py` as below. 

```python
import example

print('My_varaiable: %s' % example.cvar.My_variable)
print('fact(5): %s' % example.fact(5))
print('my_mod(7,3): %s' % example.my_mod(7,3))
print('get_time(): %s' % example.get_time())
```

How could we achieve this ? How could we convert a piece of C code into a importable python module ? What we need to do is to write an interface file, which is the input of SWIG. An interface file `example.i` might look like this:

```C
%module example

%{
#include "example.h"
%}

extern double My_variable;
extern int fact(int n);
extern int my_mod(int x, int y);
extern char *get_time();
```

where 

- `%module` is used to indicate the name of the module to be generated  
- `extern` is used to declare the interface of functions
-  `%{}%` specifies the declarations which will be needed in `example_wrap.c` file.



## Compile

First of all, we need to install some requirements, for example in Ubuntu:

```bash
sudo apt install build-essential python3-dev swig
```

Before compilation, we need to covert the `example.i` into `example_wrap.c` and `example.py` using `swig`:

```bash
swig -python example.i
```

thereafter we could use `gcc` to compile. We need to write the `Makefile` as below and simply execute `make`:

```makefile
_example.so : example.o example_wrap.o
	gcc -shared example.o example_wrap.o -o _example.so -lpython3.6

example.o : example.c
	gcc -c -fPIC -I/usr/include/python3.6 example.c

example_wrap.o : example_wrap.c
	gcc -c -fPIC -I/usr/include/python3.6 example_wrap.c

example_wrap.c example.py : example.i example.h
	swig -python example.i

clean:
	rm -f *.o *.so example_wrap.* example.py*

test:
	python3 test.py

all: _example.so test

.PHONY: clean test all

.DEFAULT_GOAL := all
```



## Setup.py File Configuration

We could also avoid writing the `Makefile`, instead we could leverage the SWIG support from `distutils` and `setup tools`. 

For example, we have the following project structure: (see [github](https://github.com/Chunpai/swig-example))

```markdown
swig-example
├── Makefile
├── setup.cfg
├── setup.py
├── src
│   |── example
│       ├── example.c
│       ├── example.h
│       ├── example.i
│       ├── __init__.py
└── tests
    ├── test_example.py
```

where `src/example` folder contains the C code we would like to wrap. Now we could configure the `setup.py` as below:

```python
from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py

EXAMPLE_EXT = Extension(
    name='_example',
    sources=[
        'src/example/example.c',
        'src/example/example.i',
    ],
)

# Build extensions before python modules,
# or the generated SWIG python files will be missing.
class BuildPy(build_py):
    def run(self):
        self.run_command('build_ext')
        super(build_py, self).run()

setup(
    name='swig-example',
    description='A Python demo for SWIG',
    version='0.0.1',
    author='Chunpai Wang',
    license='',
    author_email='chunpaiwang@gmail.com',
    url='https://chunpai.github.io',
    keywords=['SWIG'],

    packages=find_packages('src'),
    package_dir={'': 'src'},
    ext_modules=[EXAMPLE_EXT],
    cmdclass={
        'build_py': BuildPy,
    },
	
    #some dependencies or requirements
    python_requires='>=3.4',
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pytest-flake8',
    ],
    install_requires=[
        'pytest',
        'pytest-conv',
        'pytset-flake8',
    ],
    extras_require=[
    ],
    dependency_links=[
    ]
)
```







## Reference 
1. https://intermediate-and-advanced-software-carpentry.readthedocs.io/en/latest/c++-wrapping.html 
2. http://www.swig.org/Doc3.0/Python.html
3. https://note.qidong.name/2018/01/hello-swig-example/
4. https://note.qidong.name/2018/01/python-setup-requires/

















