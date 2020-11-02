//
// Created by yoloCao on 20-5-12.
//
#pragma once
#ifndef TENSORRT_YOLOV4_DARKNETADD_H
#define TENSORRT_YOLOV4_DARKNETADD_H
#pragma once
#include "plugin.hpp"
#include "serialize.hpp"
#include <vector>
#include <cuda_runtime.h>

namespace {
    constexpr const char* DARKNETADD_PLUGIN_VERSION{"001"};
    constexpr const char* DARKNETADD_PLUGIN_NAME{"DarkNetAdd"};
}

class ADDPlugin final : public onnx2trt::PluginV2 {

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
    ADDPlugin();

    ADDPlugin(void const* serialData, size_t serialLength) : _initialized(false) {
        this->deserialize(serialData, serialLength);
    }
    virtual const char* getPluginType() const override { return DARKNETADD_PLUGIN_NAME; }
    virtual const char* getPluginVersion() const override { return DARKNETADD_PLUGIN_VERSION; }
    virtual void setPluginNamespace(const char* pluginNamespace) override {}
    virtual const char* getPluginNamespace() const override { return ""; }

    virtual nvinfer1::IPluginV2* clone() const override { return new ADDPlugin{}; }

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
    ~ADDPlugin();
};

class ADDPluginCreator : public nvinfer1::IPluginCreator
{
public:
  ADDPluginCreator() {}

  ~ADDPluginCreator() {}

  const char* getPluginName() const { return DARKNETADD_PLUGIN_NAME; }

  const char* getPluginVersion() const { return DARKNETADD_PLUGIN_VERSION; }

  const nvinfer1::PluginFieldCollection* getFieldNames() { std::cerr<< "Function not implemented" << std::endl; return nullptr; }

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) { std::cerr<< "Function not implemented" << std::endl; return nullptr; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) { return new ADDPlugin{serialData, serialLength}; }

  void setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }

  const char* getPluginNamespace() const { return mNamespace.c_str(); }
private:
    std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(ADDPluginCreator);

#endif //TENSORRT_YOLOV4_DARKNETADD_H
