// Copyright (c) OpenMMLab. All rights reserved.
#include "group_points.hpp"

#include <assert.h>
#include <stdio.h>

#include <chrono>

#include "NvInferVersion.h"
#include "group_points_kernel.hpp"
#include "trt_serialize.hpp"

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"GroupPoints"};
}  // namespace

GroupPoints::GroupPoints(const std::string &name) : TRTPluginBase(name) {}

GroupPoints::GroupPoints(const std::string name, const void *data, size_t length)
    : TRTPluginBase(name) {}

nvinfer1::IPluginV2DynamicExt *GroupPoints::clone() const TRT_NOEXCEPT {
  GroupPoints *plugin = new GroupPoints(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs GroupPoints::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = inputs[1].d[1];
  ret.d[3] = inputs[1].d[2];
  return ret;
}

bool GroupPoints::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc,
                                             int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  switch (pos) {
    case 0:
      // features
      return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
              ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
    case 1:
      // idx
      return ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
             ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
    case 2:
      // output
      return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
    default:
      return true;
  }
  return true;
}

void GroupPoints::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
                                   const nvinfer1::DynamicPluginTensorDesc *outputs,
                                   int nbOutputs) TRT_NOEXCEPT {}

size_t GroupPoints::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                      const nvinfer1::PluginTensorDesc *outputs,
                                      int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

int GroupPoints::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                          const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                          void *const *outputs, void *workSpace, cudaStream_t stream) TRT_NOEXCEPT {
  int b = inputDesc[0].dims.d[0];
  int c = inputDesc[0].dims.d[1];
  int n = inputDesc[0].dims.d[2];
  int npoints = inputDesc[1].dims.d[1];
  int nsample = inputDesc[1].dims.d[2];
  const void *features = inputs[0];
  const void *indices = inputs[1];
  void *output = outputs[0];
  group_points_impl((float *)features, (int *)indices, b, c, n, npoints, nsample, (float *)output, stream);
  return 0;
}

nvinfer1::DataType GroupPoints::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                   int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *GroupPoints::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *GroupPoints::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int GroupPoints::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t GroupPoints::getSerializationSize() const TRT_NOEXCEPT { return 0; }

void GroupPoints::serialize(void *buffer) const TRT_NOEXCEPT {}

GroupPointsCreator::GroupPointsCreator() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *GroupPointsCreator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *GroupPointsCreator::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

nvinfer1::IPluginV2 *GroupPointsCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  auto *plugin = new GroupPoints(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *GroupPointsCreator::deserializePlugin(const char *name,
                                                            const void *serialData,
                                                            size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new GroupPoints(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(GroupPointsCreator);
}  // namespace mmdeploy
