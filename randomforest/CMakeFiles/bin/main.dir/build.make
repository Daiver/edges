# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_SOURCE_DIR = /home/daiver/coding/jff/sfd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/daiver/coding/jff/sfd

# Include any dependencies generated for this target.
include CMakeFiles/bin/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bin/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bin/main.dir/flags.make

CMakeFiles/bin/main.dir/main.cpp.o: CMakeFiles/bin/main.dir/flags.make
CMakeFiles/bin/main.dir/main.cpp.o: main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/daiver/coding/jff/sfd/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/bin/main.dir/main.cpp.o"
	/usr/bin/clang++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bin/main.dir/main.cpp.o -c /home/daiver/coding/jff/sfd/main.cpp

CMakeFiles/bin/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bin/main.dir/main.cpp.i"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/daiver/coding/jff/sfd/main.cpp > CMakeFiles/bin/main.dir/main.cpp.i

CMakeFiles/bin/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bin/main.dir/main.cpp.s"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/daiver/coding/jff/sfd/main.cpp -o CMakeFiles/bin/main.dir/main.cpp.s

CMakeFiles/bin/main.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/bin/main.dir/main.cpp.o.requires

CMakeFiles/bin/main.dir/main.cpp.o.provides: CMakeFiles/bin/main.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/bin/main.dir/build.make CMakeFiles/bin/main.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/bin/main.dir/main.cpp.o.provides

CMakeFiles/bin/main.dir/main.cpp.o.provides.build: CMakeFiles/bin/main.dir/main.cpp.o

CMakeFiles/bin/main.dir/decisiontree.cpp.o: CMakeFiles/bin/main.dir/flags.make
CMakeFiles/bin/main.dir/decisiontree.cpp.o: decisiontree.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/daiver/coding/jff/sfd/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/bin/main.dir/decisiontree.cpp.o"
	/usr/bin/clang++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bin/main.dir/decisiontree.cpp.o -c /home/daiver/coding/jff/sfd/decisiontree.cpp

CMakeFiles/bin/main.dir/decisiontree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bin/main.dir/decisiontree.cpp.i"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/daiver/coding/jff/sfd/decisiontree.cpp > CMakeFiles/bin/main.dir/decisiontree.cpp.i

CMakeFiles/bin/main.dir/decisiontree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bin/main.dir/decisiontree.cpp.s"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/daiver/coding/jff/sfd/decisiontree.cpp -o CMakeFiles/bin/main.dir/decisiontree.cpp.s

CMakeFiles/bin/main.dir/decisiontree.cpp.o.requires:
.PHONY : CMakeFiles/bin/main.dir/decisiontree.cpp.o.requires

CMakeFiles/bin/main.dir/decisiontree.cpp.o.provides: CMakeFiles/bin/main.dir/decisiontree.cpp.o.requires
	$(MAKE) -f CMakeFiles/bin/main.dir/build.make CMakeFiles/bin/main.dir/decisiontree.cpp.o.provides.build
.PHONY : CMakeFiles/bin/main.dir/decisiontree.cpp.o.provides

CMakeFiles/bin/main.dir/decisiontree.cpp.o.provides.build: CMakeFiles/bin/main.dir/decisiontree.cpp.o

CMakeFiles/bin/main.dir/treenode.cpp.o: CMakeFiles/bin/main.dir/flags.make
CMakeFiles/bin/main.dir/treenode.cpp.o: treenode.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/daiver/coding/jff/sfd/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/bin/main.dir/treenode.cpp.o"
	/usr/bin/clang++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bin/main.dir/treenode.cpp.o -c /home/daiver/coding/jff/sfd/treenode.cpp

CMakeFiles/bin/main.dir/treenode.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bin/main.dir/treenode.cpp.i"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/daiver/coding/jff/sfd/treenode.cpp > CMakeFiles/bin/main.dir/treenode.cpp.i

CMakeFiles/bin/main.dir/treenode.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bin/main.dir/treenode.cpp.s"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/daiver/coding/jff/sfd/treenode.cpp -o CMakeFiles/bin/main.dir/treenode.cpp.s

CMakeFiles/bin/main.dir/treenode.cpp.o.requires:
.PHONY : CMakeFiles/bin/main.dir/treenode.cpp.o.requires

CMakeFiles/bin/main.dir/treenode.cpp.o.provides: CMakeFiles/bin/main.dir/treenode.cpp.o.requires
	$(MAKE) -f CMakeFiles/bin/main.dir/build.make CMakeFiles/bin/main.dir/treenode.cpp.o.provides.build
.PHONY : CMakeFiles/bin/main.dir/treenode.cpp.o.provides

CMakeFiles/bin/main.dir/treenode.cpp.o.provides.build: CMakeFiles/bin/main.dir/treenode.cpp.o

CMakeFiles/bin/main.dir/randomforest.cpp.o: CMakeFiles/bin/main.dir/flags.make
CMakeFiles/bin/main.dir/randomforest.cpp.o: randomforest.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/daiver/coding/jff/sfd/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/bin/main.dir/randomforest.cpp.o"
	/usr/bin/clang++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bin/main.dir/randomforest.cpp.o -c /home/daiver/coding/jff/sfd/randomforest.cpp

CMakeFiles/bin/main.dir/randomforest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bin/main.dir/randomforest.cpp.i"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/daiver/coding/jff/sfd/randomforest.cpp > CMakeFiles/bin/main.dir/randomforest.cpp.i

CMakeFiles/bin/main.dir/randomforest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bin/main.dir/randomforest.cpp.s"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/daiver/coding/jff/sfd/randomforest.cpp -o CMakeFiles/bin/main.dir/randomforest.cpp.s

CMakeFiles/bin/main.dir/randomforest.cpp.o.requires:
.PHONY : CMakeFiles/bin/main.dir/randomforest.cpp.o.requires

CMakeFiles/bin/main.dir/randomforest.cpp.o.provides: CMakeFiles/bin/main.dir/randomforest.cpp.o.requires
	$(MAKE) -f CMakeFiles/bin/main.dir/build.make CMakeFiles/bin/main.dir/randomforest.cpp.o.provides.build
.PHONY : CMakeFiles/bin/main.dir/randomforest.cpp.o.provides

CMakeFiles/bin/main.dir/randomforest.cpp.o.provides.build: CMakeFiles/bin/main.dir/randomforest.cpp.o

# Object files for target bin/main
bin/main_OBJECTS = \
"CMakeFiles/bin/main.dir/main.cpp.o" \
"CMakeFiles/bin/main.dir/decisiontree.cpp.o" \
"CMakeFiles/bin/main.dir/treenode.cpp.o" \
"CMakeFiles/bin/main.dir/randomforest.cpp.o"

# External object files for target bin/main
bin/main_EXTERNAL_OBJECTS =

bin/main: CMakeFiles/bin/main.dir/main.cpp.o
bin/main: CMakeFiles/bin/main.dir/decisiontree.cpp.o
bin/main: CMakeFiles/bin/main.dir/treenode.cpp.o
bin/main: CMakeFiles/bin/main.dir/randomforest.cpp.o
bin/main: CMakeFiles/bin/main.dir/build.make
bin/main: CMakeFiles/bin/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable bin/main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bin/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bin/main.dir/build: bin/main
.PHONY : CMakeFiles/bin/main.dir/build

CMakeFiles/bin/main.dir/requires: CMakeFiles/bin/main.dir/main.cpp.o.requires
CMakeFiles/bin/main.dir/requires: CMakeFiles/bin/main.dir/decisiontree.cpp.o.requires
CMakeFiles/bin/main.dir/requires: CMakeFiles/bin/main.dir/treenode.cpp.o.requires
CMakeFiles/bin/main.dir/requires: CMakeFiles/bin/main.dir/randomforest.cpp.o.requires
.PHONY : CMakeFiles/bin/main.dir/requires

CMakeFiles/bin/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bin/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bin/main.dir/clean

CMakeFiles/bin/main.dir/depend:
	cd /home/daiver/coding/jff/sfd && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/daiver/coding/jff/sfd /home/daiver/coding/jff/sfd /home/daiver/coding/jff/sfd /home/daiver/coding/jff/sfd /home/daiver/coding/jff/sfd/CMakeFiles/bin/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bin/main.dir/depend

