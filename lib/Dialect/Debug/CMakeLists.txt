add_circt_dialect_library(CIRCTDebug
  DebugDialect.cpp
  DebugLocations.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Debug

  DEPENDS
  CIRCTDebugLocationsIncGen
  MLIRDebugIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
)
