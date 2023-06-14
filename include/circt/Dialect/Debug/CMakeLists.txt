add_circt_dialect(Debug dbg)
add_circt_dialect_doc(Debug dbg)

set(LLVM_TARGET_DEFINITIONS DebugLocations.td)
mlir_tablegen(DebugLocations.h.inc -gen-attrdef-decls)
mlir_tablegen(DebugLocations.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTDebugLocationsIncGen)
add_dependencies(circt-headers CIRCTDebugLocationsIncGen)
