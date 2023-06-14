//===- DebugLocations.td - Debug location attrs ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGLOCATIONS_TD
#define CIRCT_DIALECT_DEBUG_DEBUGLOCATIONS_TD

include "circt/Dialect/Debug/DebugDialect.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for location attributes containing debug information.
class DebugLocationAttr<string name, list<Trait> traits = []>
    : AttrDef<DebugDialect, name, traits, "::mlir::LocationAttr"> {
  let cppClassName = name;
  let mnemonic = ?;
}

def MagicLoc : DebugLocationAttr<"MagicLoc"> {
  let summary = "A named source location";
  let parameters = (ins "StringAttr":$name, "LocationAttr":$childLoc);
  let mnemonic = "magic";
  let assemblyFormat = "`<` $name `,` $childLoc `>`";
}

#endif // CIRCT_DIALECT_DEBUG_DEBUGLOCATIONS_TD
