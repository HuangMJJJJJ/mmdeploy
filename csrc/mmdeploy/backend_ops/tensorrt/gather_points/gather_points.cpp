// Copyright (c) OpenMMLab. All rights reserved.
#include "gather_points.hpp"

#include <assert.h>
#include <stdio.h>

#include <chrono>

#include "NvInferVersion.h"
#include "gather_points_kernel.hpp"
#include "trt_serialize.hpp"

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"GatherPoints"};
}  // namespace

GatherPoints::GatherPoints(const std::string &name) : TRTPluginBase(name) {}

GatherPoints::GatherPoints(const std::string name, const void *data, size_t length)
    : TRTPluginBase(name) {}

nvinfer1::IPluginV2DynamicExt *GatherPoints::clone() const TRT_NOEXCEPT {
  GatherPoints *plugin = new GatherPoints(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs GatherPoints::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  assert(inputs[0].nbDims == 3);
  assert(inputs[1].nbDims == 2);
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = inputs[1].d[1];
  return ret;
}

bool GatherPoints::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc,
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

void GatherPoints::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
                                   const nvinfer1::DynamicPluginTensorDesc *outputs,
                                   int nbOutputs) TRT_NOEXCEPT {}

size_t GatherPoints::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                      const nvinfer1::PluginTensorDesc *outputs,
                                      int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

int GatherPoints::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                          const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                          void *const *outputs, void *workSpace, cudaStream_t stream) TRT_NOEXCEPT {
  int b = inputDesc[0].dims.d[0];
  int c = inputDesc[0].dims.d[1];
  int n = inputDesc[0].dims.d[2];
  int npoints = inputDesc[1].dims.d[1];
  const void *features = inputs[0];
  const void *indices = inputs[1];
  void *output = outputs[0];
  gather_points_impl<float>((float *)features, (int *)indices, b, c, n, npoints, (float *)output, stream);
  return 0;
}

nvinfer1::DataType GatherPoints::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                   int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *GatherPoints::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *GatherPoints::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int GatherPoints::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t GatherPoints::getSerializationSize() const TRT_NOEXCEPT { return 0; }

void GatherPoints::serialize(void *buffer) const TRT_NOEXCEPT {}

GatherPointsCreator::GatherPointsCreator() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *GatherPointsCreator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *GatherPointsCreator::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

nvinfer1::IPluginV2 *GatherPointsCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  auto *plugin = new GatherPoints(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *GatherPointsCreator::deserializePlugin(const char *name,
                                                            const void *serialData,
                                                            size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new GatherPoints(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(GatherPointsCreator);
}  // namespace mmdeploy
