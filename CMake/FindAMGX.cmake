if ((NOT AMGX_DIR) OR ("${AMGX_DIR}" STREQUAL ""))
set(AMGX_DIR $ENV{AMGX_DIR} CACHE TYPE STRING)
endif ()

if(EXISTS ${AMGX_DIR}/include)
unset(AMGX_INCLUDE_PATH CACHE)
endif(EXISTS ${AMGX_DIR}/include)

find_path(AMGX_INCLUDE_PATH
    NAMES
    amgx_c.h
    PATHS
    ${AMGX_DIR}/include
)

if(EXISTS ${AMGX_DIR}/lib)
unset(AMGX_LIBRARIES_amgx CACHE)
unset(AMGX_LIBRARIES_amgxsh CACHE)
unset(AMGX_LIBRARIES_DIR CACHE)
endif(EXISTS ${AMGX_DIR}/lib)

if(WIN32)
	find_path(AMGX_LIBRARIES_DIR
		NAMES
		amgx.lib
		PATHS
		${AMGX_DIR}/lib
		)
	find_file(AMGX_LIBRARIES_amgx
		NAMES
		amgx.lib
		PATHS
		${AMGX_DIR}/lib
		)
	find_file(AMGX_LIBRARIES_amgxsh
		NAMES
		amgxsh.lib
		PATHS
		${AMGX_DIR}/lib
		)
else(WIN32)
	find_path(AMGX_LIBRARIES_DIR
		NAMES
		libamgx.a
		PATHS
		${AMGX_DIR}/lib
		)
	find_file(AMGX_LIBRARIES_amgx
		NAMES
		libamgx.a
		PATHS
		${AMGX_DIR}/lib
		)
endif(WIN32)

include(FindPackageHandleStandardArgs)
if(WIN32)
	find_package_handle_standard_args(AMGX DEFAULT_MSG AMGX_INCLUDE_PATH AMGX_LIBRARIES_DIR AMGX_LIBRARIES_amgx AMGX_LIBRARIES_amgxsh)
else(WIN32)
	find_package_handle_standard_args(AMGX DEFAULT_MSG AMGX_INCLUDE_PATH AMGX_LIBRARIES_DIR AMGX_LIBRARIES_amgx)
endif(WIN32)