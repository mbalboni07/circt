//===- PipelineOps.h - Pipeline MLIR Operations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the Pipeline ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/Pipeline/Pipeline.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace mlir;
using namespace circt;
using namespace circt::pipeline;

#include "circt/Dialect/Pipeline/PipelineDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// UnscheduledPipelineOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyPipeline(PipelineLike op) {
  bool anyInputIsAChannel = llvm::any_of(op.getInputs(), [](Value operand) {
    return operand.getType().isa<esi::ChannelType>();
  });
  bool anyOutputIsAChannel = llvm::any_of(op->getResultTypes(), [](Type type) {
    return type.isa<esi::ChannelType>();
  });

  if ((anyInputIsAChannel || anyOutputIsAChannel) &&
      !op.isLatencyInsensitive()) {
    return op.emitOpError("if any port of this pipeline is an ESI channel, all "
                          "ports must be ESI channels.");
  }

  Block *entryStage = op.getEntryStage();
  llvm::SmallVector<Type> expectedInArgTypes;
  llvm::append_range(expectedInArgTypes, op.getInputs().getTypes());
  llvm::append_range(expectedInArgTypes, op.getExtInputs().getTypes());
  size_t expectedNumArgs = expectedInArgTypes.size();
  if (entryStage->getNumArguments() != expectedNumArgs)
    return op.emitOpError("expected ")
           << expectedNumArgs << " arguments in the pipeline body block, got "
           << entryStage->getNumArguments() << ".";

  for (size_t i = 0; i < expectedNumArgs; i++) {
    Type expectedInArg = expectedInArgTypes[i];
    Type bodyArg = entryStage->getArgument(i).getType();

    if (op.isLatencyInsensitive())
      expectedInArg = expectedInArg.cast<esi::ChannelType>().getInner();

    if (expectedInArg != bodyArg)
      return op.emitOpError("expected body block argument ")
             << i << " to have type " << expectedInArg << ", got " << bodyArg
             << ".";
  }

  return success();
}

LogicalResult UnscheduledPipelineOp::verify() { return verifyPipeline(*this); }

bool UnscheduledPipelineOp::isLatencyInsensitive() {
  bool allInputsAreChannels = llvm::all_of(getInputs(), [](Value operand) {
    return operand.getType().isa<esi::ChannelType>();
  });
  bool allOutputsAreChannels = llvm::all_of(
      getResultTypes(), [](Type type) { return type.isa<esi::ChannelType>(); });
  return allInputsAreChannels && allOutputsAreChannels;
}

//===----------------------------------------------------------------------===//
// ScheduledPipelineOp
//===----------------------------------------------------------------------===//

void ScheduledPipelineOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                TypeRange results, ValueRange inputs,
                                ValueRange extInputs, Value clock,
                                Value reset) {
  odsState.addOperands(inputs);
  odsState.addOperands(extInputs);
  odsState.addOperands(clock);
  odsState.addOperands(reset);
  odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()),
                                       static_cast<int32_t>(extInputs.size()),
                                       static_cast<int32_t>(1),
                                       static_cast<int32_t>(1)}));
  auto *region = odsState.addRegion();
  odsState.addTypes(results);

  // Add the entry stage
  auto &entryBlock = region->emplaceBlock();
  llvm::SmallVector<Location> entryArgLocs(inputs.size(), odsState.location);
  entryBlock.addArguments(
      inputs.getTypes(),
      llvm::SmallVector<Location>(inputs.size(), odsState.location));
  entryBlock.addArguments(
      extInputs.getTypes(),
      llvm::SmallVector<Location>(extInputs.size(), odsState.location));
}

Block *ScheduledPipelineOp::addStage() {
  OpBuilder builder(getContext());
  Block *stage = builder.createBlock(&getRegion());
  return stage;
}

llvm::SmallVector<Block *> ScheduledPipelineOp::getOrderedStages() {
  Block *currentStage = getEntryStage();
  llvm::SmallVector<Block *> orderedStages;
  do {
    orderedStages.push_back(currentStage);
    if (auto stageOp = dyn_cast<StageOp>(currentStage->getTerminator()))
      currentStage = stageOp.getNextStage();
    else
      currentStage = nullptr;
  } while (currentStage);

  return orderedStages;
}

Block *ScheduledPipelineOp::getLastStage() { return getOrderedStages().back(); }

bool ScheduledPipelineOp::isMaterialized() {
  // We determine materialization as if any pipeline stage has an explicit
  // input.
  return llvm::any_of(getStages(), [this](Block &block) {
    // The entry stage doesn't count since it'll always have arguments.
    if (&block == getEntryStage())
      return false;
    return block.getNumArguments() != 0;
  });
}

LogicalResult ScheduledPipelineOp::verify() {
  if (failed(verifyPipeline(*this)))
    return failure();

  // Cache external inputs in a set for fast lookup.
  llvm::DenseSet<Value> extInputs;
  for (auto extInput : getInnerExtInputs())
    extInputs.insert(extInput);

  // Phase invariant - if any block has arguments, we
  bool materialized = isMaterialized();
  if (materialized) {
    // Check that all values used within a stage are defined within the stage.
    for (auto &stage : getStages()) {
      for (auto &op : stage) {
        for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
          bool err = false;
          if (auto *definingOp = operand.getDefiningOp()) {
            // Constants are allowed to be used across stages.
            if (definingOp->hasTrait<OpTrait::ConstantLike>())
              continue;
            err = definingOp->getBlock() != &stage;
          } else if (extInputs.contains(operand)) {
            // This is an external input; legal to reference everywhere.
            continue;
          } else {
            // This is a block argument;
            err = !llvm::is_contained(stage.getArguments(), operand);
          }

          if (err)
            return op.emitOpError(
                       "Pipeline is in register materialized mode - operand ")
                   << index
                   << " is defined in a different stage, which is illegal.";
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  Operation *parent = getOperation()->getParentOp();
  size_t nInputs = getInputs().size();
  size_t nResults = parent->getNumResults();
  if (nInputs != nResults)
    return emitOpError("expected ")
           << nResults << " return values, got " << nInputs << ".";

  for (auto [inType, reqType] :
       llvm::zip(getInputs().getTypes(), parent->getResultTypes())) {
    if (inType != reqType)
      return emitOpError("expected return value of type ")
             << reqType << ", got " << inType << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StageOp
//===----------------------------------------------------------------------===//

SuccessorOperands StageOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  // Successor operands are everything but the "valid" input to this stage op.
  // Places a hard assumption on the regs and passthrough operands being next to
  // each other in the operand list.
  auto mutableRange =
      mlir::MutableOperandRange(getOperation(), 0, getNumOperands() - 1);
  return SuccessorOperands(mutableRange);
}

Block *StageOp::getSuccessorForOperands(ArrayRef<Attribute>) {
  return getNextStage();
}

LogicalResult StageOp::verify() {
  // Verify that the target block has the correct arguments as this stage op.
  llvm::SmallVector<Type> expectedTargetArgTypes;
  llvm::append_range(expectedTargetArgTypes, getRegisters().getTypes());
  llvm::append_range(expectedTargetArgTypes, getPassthroughs().getTypes());
  Block *targetStage = getNextStage();

  if (targetStage->getNumArguments() != expectedTargetArgTypes.size())
    return emitOpError("expected ") << expectedTargetArgTypes.size()
                                    << " arguments in the target stage, got "
                                    << targetStage->getNumArguments() << ".";

  for (auto [index, it] : llvm::enumerate(llvm::zip(
           expectedTargetArgTypes, targetStage->getArgumentTypes()))) {
    auto [arg, barg] = it;
    if (arg != barg)
      return emitOpError("expected target stage argument ")
             << index << " to have type " << arg << ", got " << barg << ".";
  }

  return success();
}

#include "circt/Dialect/Pipeline/PipelineInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"

void PipelineDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"
      >();
}
