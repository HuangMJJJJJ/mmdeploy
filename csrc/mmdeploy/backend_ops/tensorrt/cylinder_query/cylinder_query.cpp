// Copyright (c) OpenMMLab. All rights reserved.
#include "cylinder_query.hpp"

#include <assert.h>
#include <stdio.h>

#include <chrono>

#include "NvInferVersion.h"
#include "cylinder_query_kernel.hpp"
#include "trt_serialize.hpp"

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"CylinderQuery"};
}  // namespace

CylinderQuery::CylinderQuery(const std::string &name, float hmax, float hmin, int nsample, float radius)
    : TRTPluginBase(name), mHmax(hmax), mHmin(hmin), mNsample(nsample), mRadius(radius) {}

CylinderQuery::CylinderQuery(const std::string name, const void *data,
                                             size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &mHmax);
  deserialize_value(&data, &length, &mHmin);
  deserialize_value(&data, &length, &mNsample);
  deserialize_value(&data, &length, &mRadius);
}

nvinfer1::IPluginV2DynamicExt *CylinderQuery::clone() const TRT_NOEXCEPT {
  CylinderQuery *plugin = new CylinderQuery(mLayerName, mHmax, mHmin, mNsample, mRadius);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::DimsExprs CylinderQuery::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  auto B = inputs[1].d[0];
  auto npoint = inputs[1].d[1];
  nvinfer1::DimsExprs ret{3, {B, npoint, exprBuilder.constant(mNsample)}};
  return ret;
}

bool CylinderQuery::supportsFormatCombination(int pos,
                                                      const nvinfer1::PluginTensorDesc *ioDesc,
                                                      int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  switch (pos) {
    case 0:
      // xyz
      return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
              ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
    case 1:
      // new_xyz
      return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
              ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
    case 2:
      // rot
      return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
              ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
    case 3:
      // output
      return ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
             ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
    default:
      return true;
  }
  return true;
}

void CylinderQuery::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs,
                                            int nbInputs,
                                            const nvinfer1::DynamicPluginTensorDesc *outputs,
                                            int nbOutputs) TRT_NOEXCEPT {}

size_t CylinderQuery::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                               int nbInputs,
                                               const nvinfer1::PluginTensorDesc *outputs,
                                               int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

int CylinderQuery::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs, void *const *outputs, void *workSpace,
                                   cudaStream_t stream) TRT_NOEXCEPT {
  const void *xyz = inputs[0];
  const void *new_xyz = inputs[1];
  const void *rot = inputs[2];
  const void *idx = outputs[0];

  int b = inputDesc[0].dims.d[0];
  int n = inputDesc[0].dims.d[1];
  int m = inputDesc[1].dims.d[1];
  

  query_cylinder_point_impl(b, n, m, mRadius, mHmin, mHmax, mNsample, (float *) new_xyz, (float *) xyz, (float *) rot, (int *)idx, stream);
  return 0;
}

nvinfer1::DataType CylinderQuery::getOutputDataType(int index,
                                                            const nvinfer1::DataType *inputTypes,
                                                            int nbInputs) const TRT_NOEXCEPT {
  return nvinfer1::DataType::kINT32;
}

// IPluginV2 Methods
const char *CylinderQuery::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *CylinderQuery::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int CylinderQuery::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t CylinderQuery::getSerializationSize() const TRT_NOEXCEPT {
  return serialized_size(mHmax) + serialized_size(mHmin) + serialized_size(mNsample) + serialized_size(mRadius);
}

void CylinderQuery::serialize(void *buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, mHmax);
  serialize_value(&buffer, mHmin);
  serialize_value(&buffer, mNsample);
  serialize_value(&buffer, mRadius);
}

CylinderQueryCreator::CylinderQueryCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(nvinfer1::PluginField("hmax"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("hmin"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("nsample"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("radius"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *CylinderQueryCreator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *CylinderQueryCreator::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2 *CylinderQueryCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  float radius = -1.0;
  float hmin = -1.0;
  float hmax = -1.0;
  int nsample = -1;
  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("radius") == 0) {
      radius = static_cast<const float *>(fc->fields[i].data)[0];
    }
    else if (field_name.compare("hmin") == 0) {
      hmin = static_cast<const float *>(fc->fields[i].data)[0];
    }
    else if (field_name.compare("hmax") == 0) {
      hmax = static_cast<const float *>(fc->fields[i].data)[0];
    }
    else if (field_name.compare("nsample") == 0) {
      nsample = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }
  assert(radius != -1.0);
  assert(hmin != -1.0);
  assert(hmax != -1.0);
  assert(nsample != -1);
  CylinderQuery *plugin = new CylinderQuery(name, hmax, hmin, nsample, radius);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *CylinderQueryCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new CylinderQuery(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(CylinderQueryCreator);
}  // namespace mmdeploy
