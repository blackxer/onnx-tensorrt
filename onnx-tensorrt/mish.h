//
// Created by yoloCao on 20-5-11.
//

#ifndef TENSORRT_YOLOV4_MISH_H
#define TENSORRT_YOLOV4_MISH_H

#pragma once
#include "plugin.hpp"
#include "serialize.hpp"
#include <vector>
#include <cuda_runtime.h>

namespace {
    constexpr const char* MISH_PLUGIN_VERSION{"001"};
    constexpr const char* MISH_PLUGIN_NAME{"Mish"};
}

class MishPlugin final : public onnx2trt::PluginV2 {

    bool _initialized;
protected:
    void deserialize(void const* serialData, size_t serialLength) {
        deserializeBase(serialData, serialLength);
    }
    size_t getSerializationSize() const override {
        return  getBaseSerializationSize();
    }
    void serialize(void *buffer) const override {
        serializeBase(buffer);
    }
public:
    MishPlugin();

    MishPlugin(void const* serialData, size_t serialLength) : _initialized(false) {
        this->deserialize(serialData, serialLength);
    }
    const char* getPluginType() const override { return MISH_PLUGIN_NAME; }
    virtual const char* getPluginVersion() const override { return MISH_PLUGIN_VERSION; }
    virtual void setPluginNamespace(const char* pluginNamespace) override {}
    virtual const char* getPluginNamespace() const override { return ""; }

    virtual nvinfer1::IPluginV2* clone() const override { return new MishPlugin{}; }

    int getNbOutputs() const override { return 1; }
    nvinfer1::Dims getOutputDimensions(int index,
                                       const nvinfer1::Dims *inputDims,
                                       int nbInputs) override;
    int initialize() override;
    void terminate() override;
    int enqueue(int batchSize,
                const void *const *inputs, void **outputs,
                void *workspace, cudaStream_t stream) override;
    size_t getWorkspaceSize(int maxBatchSize) const override;
    ~MishPlugin();
};

class MishPluginCreator : public nvinfer1::IPluginCreator
{
public:
  MishPluginCreator() {}

  ~MishPluginCreator() {}

  const char* getPluginName() const { return MISH_PLUGIN_NAME; }

  const char* getPluginVersion() const { return MISH_PLUGIN_VERSION; }

  const nvinfer1::PluginFieldCollection* getFieldNames() { std::cerr<< "Function not implemented" << std::endl; return nullptr; }

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) { std::cerr<< "Function not implemented" << std::endl; return nullptr; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) { return new MishPlugin{serialData, serialLength}; }

  void setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }

  const char* getPluginNamespace() const { return mNamespace.c_str(); }
private:
    std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(MishPluginCreator);

#endif //TENSORRT_YOLOV4_MISH_H
