CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

OPTION(Debug "Set to ON to compile with debug symbols" OFF)
OPTION(G "Set to ON to compile with optimisations and debug symbols" OFF)

PROJECT(compare_MD)

SET(CMAKE_BUILD_TYPE Release)

IF(Debug)
	SET(CMAKE_BUILD_TYPE Debug)
	MESSAGE(STATUS "Compiling in Debug mode")
ENDIF(Debug)

IF(G)
	SET(CMAKE_BUILD_TYPE RelWithDebInfo)
	MESSAGE(STATUS "Compiling in Release+Debug symbols mode")
ENDIF(G)

IF(NOT CMAKE_BUILD_TYPE MATCHES Debug)
	IF(INTEL)
		ADD_DEFINITIONS(-fast)
	ELSE()
		ADD_DEFINITIONS(-ffast-math)
	ENDIF(INTEL)
ENDIF()

# return the current date as day/month/year
MACRO(CURRENT_DATE result)
    IF(WIN32)
        EXECUTE_PROCESS(COMMAND "cmd" " /C date /T" OUTPUT_VARIABLE ${result})
        string(REGEX REPLACE "(..)/(..)/..(..).*" "\\1/\\2/\\3" ${result} ${${result}})
    ELSEIF(UNIX)
        EXECUTE_PROCESS(COMMAND "date" "+%d/%m/%Y" OUTPUT_VARIABLE ${result})
        string(REGEX REPLACE "(..)/(..)/..(..).*" "\\1/\\2/\\3" ${result} ${${result}})
    ELSE(WIN32)
        MESSAGE(SEND_ERROR "date not implemented")
        SET(${result} 000000)
    ENDIF (WIN32)
ENDMACRO(CURRENT_DATE)

CURRENT_DATE(today)
ADD_DEFINITIONS(-Wall -Wshadow)
ADD_DEFINITIONS(-D'BUILD_TIME="${today}"')

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

SET(compare_MD_SOURCES
	src/main.cpp
	src/CPU.cpp
	src/utils.cpp
)

ADD_EXECUTABLE(compare_MD ${compare_MD_SOURCES})

FIND_PACKAGE(CUDA)
IF(CUDA_FOUND)
	# this is required when working with CUDA and newer gcc versions (> 5.4)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

	SET(compare_MD_CUDA_SOURCES
	src/CUDA.cu
	)

	CUDA_ADD_EXECUTABLE(compare_MD_CUDA ${compare_MD_SOURCES} ${compare_MD_CUDA_SOURCES})
	TARGET_COMPILE_DEFINITIONS(compare_MD_CUDA PUBLIC CUDA)
ENDIF(CUDA_FOUND)
