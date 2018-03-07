set(libtest SDLMadness)
set(test_name ${libtest}Test)
xpGetExtern(externIncs externLibs PUBLIC boost)
###################
# source files
set(${test_name}_srcs
  SDLMadnessTestsMain.cpp
  )
source_group("" FILES ${${test_name}_srcs})
#######################################
# test executable
add_executable(${test_name} ${${test_name}_srcs})
target_link_libraries(${test_name} ${libtest} ${externLibs})
target_include_directories(${test_name} ${externIncs})
add_test(NAME ${test_name} COMMAND ${test_name})
xpPostBuildCopy(${test_name} ${CMAKE_CURRENT_SOURCE_DIR}/four_team_test.csv .)
