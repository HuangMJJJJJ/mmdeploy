// Copyright (c) OpenMMLab. All rights reserved.
#include "furthest_point_sampling.hpp"

#include <assert.h>
#include <stdio.h>

#include <chrono>

#include "NvInferVersion.h"
#include "furthest_point_sampling_kernel.hpp"
#include "trt_serialize.hpp"

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"FurthestPointSampling"};
}  // namespace
FurthestPointSampling::FurthestPointSampling(const std::string &name, int npoint)
    : TRTPluginBase(name), mNpoint(npoint) {}

FurthestPointSampling::FurthestPointSampling(const std::string name, const void *data,
                                             size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &mNpoint);
}

nvinfer1::IPluginV2DynamicExt *FurthestPointSampling::clone() const TRT_NOEXCEPT {
  FurthestPointSampling *plugin = new FurthestPointSampling(mLayerName, mNpoint);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::DimsExprs FurthestPointSampling::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs ret{2, {inputs[0].d[0], exprBuilder.constant(mNpoint)}};
  return ret;
}

bool FurthestPointSampling::supportsFormatCombination(int pos,
                                                      const nvinfer1::PluginTensorDesc *ioDesc,
                                                      int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  switch (pos) {
    case 0:
      // xyz
      return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
              ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
    case 1:
      // output
      return ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
             ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
    default:
      return true;
  }
  return true;
}

void FurthestPointSampling::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs,
                                            int nbInputs,
                                            const nvinfer1::DynamicPluginTensorDesc *outputs,
                                            int nbOutputs) TRT_NOEXCEPT {}

size_t FurthestPointSampling::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                               int nbInputs,
                                               const nvinfer1::PluginTensorDesc *outputs,
                                               int nbOutputs) const TRT_NOEXCEPT {
  int b = inputs[0].dims.d[0];
  int n = inputs[0].dims.d[1];
  auto sizeof_float = mmdeploy::getElementSize(inputs[0].type);
  return mmdeploy::getAlignedSize(sizeof_float * n * b);
}

int FurthestPointSampling::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs, void *const *outputs, void *workSpace,
                                   cudaStream_t stream) TRT_NOEXCEPT {
  const void *xyz = inputs[0];
  auto dtype = inputDesc[0].type;
  auto sizeof_float = mmdeploy::getElementSize(dtype);
  const void *idxs = outputs[0];
  int b = inputDesc[0].dims.d[0];
  int n = inputDesc[0].dims.d[1];
  int m = mNpoint;
  float temp[mmdeploy::getAlignedSize(sizeof_float * n * b)];
  for(size_t i = 0; i < mmdeploy::getAlignedSize(sizeof_float * n * b); i++){
    temp[i] = 1e10;
  }
  cudaMemcpyAsync(workSpace, temp, mmdeploy::getAlignedSize(sizeof_float * n * b), cudaMemcpyHostToDevice);
  furthest_point_sampling_impl(b, n, m, (float *)xyz,(float *)workSpace, (int *)idxs, stream);
  return 0;
}

nvinfer1::DataType FurthestPointSampling::getOutputDataType(int index,
                                                            const nvinfer1::DataType *inputTypes,
                                                            int nbInputs) const TRT_NOEXCEPT {
  return nvinfer1::DataType::kINT32;
}

// IPluginV2 Methods
const char *FurthestPointSampling::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *FurthestPointSampling::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int FurthestPointSampling::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t FurthestPointSampling::getSerializationSize() const TRT_NOEXCEPT {
  return serialized_size(mNpoint);
}

void FurthestPointSampling::serialize(void *buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, mNpoint);
}

FurthestPointSamplingCreator::FurthestPointSamplingCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(nvinfer1::PluginField("npoint"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *FurthestPointSamplingCreator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *FurthestPointSamplingCreator::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2 *FurthestPointSamplingCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  int npoint = -1;
  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("npoint") == 0) {
      npoint = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }
  assert(npoint != -1);
  FurthestPointSampling *plugin = new FurthestPointSampling(name, npoint);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *FurthestPointSamplingCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new FurthestPointSampling(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(FurthestPointSamplingCreator);
}  // namespace mmdeploy
