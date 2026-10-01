# The shared decision engine: linked by llama-server's /decision endpoint. test-decision-engine
# compiles the engine sources directly, so a tests-only or LLAMA_BUILD_SERVER=OFF build must not
# compile this library.
if (LLAMA_BUILD_SERVER)
    add_library(llama-decision STATIC
            decision-engine.cpp decision-engine.h
            decision-protocol.cpp decision-protocol.h
            generic_frontend.cpp generic_frontend.h
            labels.cpp labels.h
            letter_readout.cpp letter_readout.h)
    target_include_directories(llama-decision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(llama-decision PUBLIC llama-common llama)
    target_compile_features(llama-decision PUBLIC cxx_std_17)
    # llama-server links it into a shared library (libllama-server-impl)
    set_target_properties(llama-decision PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
