//===- DropConst.cpp - Check and remove const types -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DropConst pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Iterators.h"

using namespace circt;
using namespace firrtl;

/// Returns null type if no conversion is needed.
static FIRRTLBaseType convertType(FIRRTLBaseType type) {
  auto nonConstType = type.getAllConstDroppedType();
  return nonConstType != type ? nonConstType : FIRRTLBaseType{};
}

/// Returns null type if no conversion is needed.
static Type convertType(Type type) {
  if (auto base = type.dyn_cast<FIRRTLBaseType>()) {
    return convertType(base);
  }

  if (auto refType = type.dyn_cast<RefType>()) {
    if (auto converted = convertType(refType.getType()))
      return RefType::get(converted, refType.getForceable());
  }

  return {};
}

namespace {
class DropConstPass : public DropConstBase<DropConstPass> {
  void runOnOperation() override {
    auto module = getOperation();

    // Convert the module body if present
    auto walkResult =
        module->walk<mlir::WalkOrder::PostOrder,
                     mlir::ReverseIterator>([](Operation *op) -> WalkResult {
          if (auto constCastOp = dyn_cast<ConstCastOp>(op)) {
            // Remove any `ConstCastOp`, replacing results with inputs
            constCastOp.getResult().replaceAllUsesWith(constCastOp.getInput());
            constCastOp->erase();
            return WalkResult::advance();
          }

          // Check that any `RegResetOp` with an async reset has a 'const' reset
          // value
          if (auto regResetOp = dyn_cast<RegResetOp>(op)) {
            if (regResetOp.getResetSignal().getType().isa<AsyncResetType>() &&
                !regResetOp.getResetValue().getType().isConst()) {
              regResetOp.emitError()
                  << "register " << regResetOp.getNameAttr()
                  << " has an async reset, but its reset value is not 'const'";
              return WalkResult::interrupt();
            }
          }

          // Convert any block arguments
          for (auto &region : op->getRegions())
            for (auto &block : region.getBlocks())
              for (auto argument : block.getArguments())
                if (auto convertedType = convertType(argument.getType()))
                  argument.setType(convertedType);

          for (auto result : op->getResults())
            if (auto convertedType = convertType(result.getType()))
              result.setType(convertedType);

          return WalkResult::advance();
        });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Update the module signature with non-'const' ports
    SmallVector<Attribute> portTypes;
    portTypes.reserve(module.getNumPorts());
    bool convertedAny = false;
    llvm::transform(module.getPortTypes(), std::back_inserter(portTypes),
                    [&](Attribute type) -> Attribute {
                      if (auto convertedType =
                              convertType(type.cast<TypeAttr>().getValue())) {
                        convertedAny = true;
                        return TypeAttr::get(convertedType);
                      }
                      return type;
                    });
    if (convertedAny)
      module->setAttr(FModuleLike::getPortTypesAttrName(),
                      ArrayAttr::get(module.getContext(), portTypes));
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createDropConstPass() {
  return std::make_unique<DropConstPass>();
}
