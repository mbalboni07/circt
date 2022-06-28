//===- OFIRTypes.cpp - Implement the OFIR dialect type system ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the OFIR dialect type system.
//
//===----------------------------------------------------------------------===//

//#include "circt/Dialect/OFIR/OFIRTypes.h"
//#include "circt/Dialect/OFIR/OFIROps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

//using namespace circt;
//using namespace ofir;

using mlir::OptionalParseResult;
using mlir::TypeStorageAllocator;

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation for types.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/OFIR/OFIRTypes.cpp.inc"

