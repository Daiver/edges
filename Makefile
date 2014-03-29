# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/daiver/coding/edges

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/daiver/coding/edges

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running interactive CMake command-line interface..."
	/usr/bin/cmake -i .
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/daiver/coding/edges/CMakeFiles /home/daiver/coding/edges/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/daiver/coding/edges/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named bin/activate

# Build rule for target.
bin/activate: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 bin/activate
.PHONY : bin/activate

# fast build rule for target.
bin/activate/fast:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/build
.PHONY : bin/activate/fast

#=============================================================================
# Target rules for targets named bin/discretize_test

# Build rule for target.
bin/discretize_test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 bin/discretize_test
.PHONY : bin/discretize_test

# fast build rule for target.
bin/discretize_test/fast:
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/build
.PHONY : bin/discretize_test/fast

#=============================================================================
# Target rules for targets named bin/test2

# Build rule for target.
bin/test2: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 bin/test2
.PHONY : bin/test2

# fast build rule for target.
bin/test2/fast:
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/build
.PHONY : bin/test2/fast

activate.o: activate.cpp.o
.PHONY : activate.o

# target to build an object file
activate.cpp.o:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/activate.cpp.o
.PHONY : activate.cpp.o

activate.i: activate.cpp.i
.PHONY : activate.i

# target to preprocess a source file
activate.cpp.i:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/activate.cpp.i
.PHONY : activate.cpp.i

activate.s: activate.cpp.s
.PHONY : activate.s

# target to generate assembly for a file
activate.cpp.s:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/activate.cpp.s
.PHONY : activate.cpp.s

common.o: common.cpp.o
.PHONY : common.o

# target to build an object file
common.cpp.o:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/common.cpp.o
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/common.cpp.o
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/common.cpp.o
.PHONY : common.cpp.o

common.i: common.cpp.i
.PHONY : common.i

# target to preprocess a source file
common.cpp.i:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/common.cpp.i
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/common.cpp.i
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/common.cpp.i
.PHONY : common.cpp.i

common.s: common.cpp.s
.PHONY : common.s

# target to generate assembly for a file
common.cpp.s:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/common.cpp.s
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/common.cpp.s
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/common.cpp.s
.PHONY : common.cpp.s

decisiontree.o: decisiontree.cpp.o
.PHONY : decisiontree.o

# target to build an object file
decisiontree.cpp.o:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/decisiontree.cpp.o
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/decisiontree.cpp.o
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/decisiontree.cpp.o
.PHONY : decisiontree.cpp.o

decisiontree.i: decisiontree.cpp.i
.PHONY : decisiontree.i

# target to preprocess a source file
decisiontree.cpp.i:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/decisiontree.cpp.i
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/decisiontree.cpp.i
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/decisiontree.cpp.i
.PHONY : decisiontree.cpp.i

decisiontree.s: decisiontree.cpp.s
.PHONY : decisiontree.s

# target to generate assembly for a file
decisiontree.cpp.s:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/decisiontree.cpp.s
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/decisiontree.cpp.s
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/decisiontree.cpp.s
.PHONY : decisiontree.cpp.s

desc.o: desc.cpp.o
.PHONY : desc.o

# target to build an object file
desc.cpp.o:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/desc.cpp.o
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/desc.cpp.o
.PHONY : desc.cpp.o

desc.i: desc.cpp.i
.PHONY : desc.i

# target to preprocess a source file
desc.cpp.i:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/desc.cpp.i
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/desc.cpp.i
.PHONY : desc.cpp.i

desc.s: desc.cpp.s
.PHONY : desc.s

# target to generate assembly for a file
desc.cpp.s:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/desc.cpp.s
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/desc.cpp.s
.PHONY : desc.cpp.s

discretize.o: discretize.cpp.o
.PHONY : discretize.o

# target to build an object file
discretize.cpp.o:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/discretize.cpp.o
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/discretize.cpp.o
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/discretize.cpp.o
.PHONY : discretize.cpp.o

discretize.i: discretize.cpp.i
.PHONY : discretize.i

# target to preprocess a source file
discretize.cpp.i:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/discretize.cpp.i
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/discretize.cpp.i
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/discretize.cpp.i
.PHONY : discretize.cpp.i

discretize.s: discretize.cpp.s
.PHONY : discretize.s

# target to generate assembly for a file
discretize.cpp.s:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/discretize.cpp.s
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/discretize.cpp.s
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/discretize.cpp.s
.PHONY : discretize.cpp.s

discretize_test.o: discretize_test.cpp.o
.PHONY : discretize_test.o

# target to build an object file
discretize_test.cpp.o:
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/discretize_test.cpp.o
.PHONY : discretize_test.cpp.o

discretize_test.i: discretize_test.cpp.i
.PHONY : discretize_test.i

# target to preprocess a source file
discretize_test.cpp.i:
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/discretize_test.cpp.i
.PHONY : discretize_test.cpp.i

discretize_test.s: discretize_test.cpp.s
.PHONY : discretize_test.s

# target to generate assembly for a file
discretize_test.cpp.s:
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/discretize_test.cpp.s
.PHONY : discretize_test.cpp.s

randomforest.o: randomforest.cpp.o
.PHONY : randomforest.o

# target to build an object file
randomforest.cpp.o:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/randomforest.cpp.o
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/randomforest.cpp.o
.PHONY : randomforest.cpp.o

randomforest.i: randomforest.cpp.i
.PHONY : randomforest.i

# target to preprocess a source file
randomforest.cpp.i:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/randomforest.cpp.i
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/randomforest.cpp.i
.PHONY : randomforest.cpp.i

randomforest.s: randomforest.cpp.s
.PHONY : randomforest.s

# target to generate assembly for a file
randomforest.cpp.s:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/randomforest.cpp.s
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/randomforest.cpp.s
.PHONY : randomforest.cpp.s

test2.o: test2.cpp.o
.PHONY : test2.o

# target to build an object file
test2.cpp.o:
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/test2.cpp.o
.PHONY : test2.cpp.o

test2.i: test2.cpp.i
.PHONY : test2.i

# target to preprocess a source file
test2.cpp.i:
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/test2.cpp.i
.PHONY : test2.cpp.i

test2.s: test2.cpp.s
.PHONY : test2.s

# target to generate assembly for a file
test2.cpp.s:
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/test2.cpp.s
.PHONY : test2.cpp.s

treenode.o: treenode.cpp.o
.PHONY : treenode.o

# target to build an object file
treenode.cpp.o:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/treenode.cpp.o
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/treenode.cpp.o
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/treenode.cpp.o
.PHONY : treenode.cpp.o

treenode.i: treenode.cpp.i
.PHONY : treenode.i

# target to preprocess a source file
treenode.cpp.i:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/treenode.cpp.i
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/treenode.cpp.i
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/treenode.cpp.i
.PHONY : treenode.cpp.i

treenode.s: treenode.cpp.s
.PHONY : treenode.s

# target to generate assembly for a file
treenode.cpp.s:
	$(MAKE) -f CMakeFiles/bin/activate.dir/build.make CMakeFiles/bin/activate.dir/treenode.cpp.s
	$(MAKE) -f CMakeFiles/bin/discretize_test.dir/build.make CMakeFiles/bin/discretize_test.dir/treenode.cpp.s
	$(MAKE) -f CMakeFiles/bin/test2.dir/build.make CMakeFiles/bin/test2.dir/treenode.cpp.s
.PHONY : treenode.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... bin/activate"
	@echo "... bin/discretize_test"
	@echo "... bin/test2"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... activate.o"
	@echo "... activate.i"
	@echo "... activate.s"
	@echo "... common.o"
	@echo "... common.i"
	@echo "... common.s"
	@echo "... decisiontree.o"
	@echo "... decisiontree.i"
	@echo "... decisiontree.s"
	@echo "... desc.o"
	@echo "... desc.i"
	@echo "... desc.s"
	@echo "... discretize.o"
	@echo "... discretize.i"
	@echo "... discretize.s"
	@echo "... discretize_test.o"
	@echo "... discretize_test.i"
	@echo "... discretize_test.s"
	@echo "... randomforest.o"
	@echo "... randomforest.i"
	@echo "... randomforest.s"
	@echo "... test2.o"
	@echo "... test2.i"
	@echo "... test2.s"
	@echo "... treenode.o"
	@echo "... treenode.i"
	@echo "... treenode.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

