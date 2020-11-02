//
// Created by cao on 19-12-20.
//

#ifndef ONNX2TRT_DCNV2_H
#define ONNX2TRT_DCNV2_H

#pragma once

#include <NvInfer.h>
#include "plugin.hpp"
#include "serialize.hpp"
#include <cudnn.h>
#include <vector>
#include <cublas_v2.h>
#include <cuda.h>

namespace {
    constexpr const char* DCNV2_PLUGIN_VERSION{"001"};
    constexpr const char* DCNV2_PLUGIN_NAME{"DCNv2"};
}

class DCNv2Plugin final : public onnx2trt::PluginV2 {
    int _in_channel;
    int _out_channel;
    int _kernel_H;
    int _kernel_W;
    int _deformable_group;
    int _dilation;
    int _groups; // not use
    int _padding;
    int _stride;
    std::vector<float> _h_weight;
    std::vector<float> _h_bias;
    float* _d_weight;
    float* _d_bias;
    float* _d_ones;
    float *_d_columns;
    nvinfer1::Weights _weight, _bias;

    bool _initialized;

protected:
    void deserialize(void const* serialData, size_t serialLength) {
        deserializeBase(serialData, serialLength);
        deserialize_value(&serialData, &serialLength, &_in_channel);
        deserialize_value(&serialData, &serialLength, &_out_channel);
        deserialize_value(&serialData, &serialLength, &_kernel_H);
        deserialize_value(&serialData, &serialLength, &_kernel_W);
        deserialize_value(&serialData, &serialLength, &_deformable_group);
        deserialize_value(&serialData, &serialLength, &_dilation);
        deserialize_value(&serialData, &serialLength, &_groups);
        deserialize_value(&serialData, &serialLength, &_padding);
        deserialize_value(&serialData, &serialLength, &_stride);
        deserialize_value(&serialData, &serialLength, &_h_weight);
        deserialize_value(&serialData, &serialLength, &_h_bias);
    }
    size_t getSerializationSize() const override {
        return (serialized_size(_in_channel) +
                serialized_size(_out_channel) +
                serialized_size(_kernel_H) +
                serialized_size(_kernel_W) +
                serialized_size(_deformable_group) +
                serialized_size(_dilation) +
                serialized_size(_groups) +
                serialized_size(_padding) +
                serialized_size(_stride) +
                serialized_size(_h_weight) +
                serialized_size(_h_bias)
               ) + getBaseSerializationSize();
    }
    void serialize(void *buffer) const override {
        serializeBase(buffer);
        serialize_value(&buffer, _in_channel);
        serialize_value(&buffer, _out_channel);
        serialize_value(&buffer, _kernel_H);
        serialize_value(&buffer, _kernel_W);
        serialize_value(&buffer, _deformable_group);
        serialize_value(&buffer, _dilation);
        serialize_value(&buffer, _groups);
        serialize_value(&buffer, _padding);
        serialize_value(&buffer, _stride);
        serialize_value(&buffer, _h_weight);
        serialize_value(&buffer, _h_bias);
    }
public:
    DCNv2Plugin(int in_channel,
                int out_channel,
                int kernel_H,
                int kernel_W,
                int deformable_group,
                int dilation,
                int groups,
                int padding,
                int stride,
                nvinfer1::Weights const& weight,
                nvinfer1::Weights const& bias);

    DCNv2Plugin(void const* serialData, size_t serialLength) : _initialized(false) {
        this->deserialize(serialData, serialLength);
    }

    virtual const char* getPluginType() const override { return DCNV2_PLUGIN_NAME; }
    virtual const char* getPluginVersion() const override { return DCNV2_PLUGIN_VERSION; }
    virtual void setPluginNamespace(const char* pluginNamespace) override {}
    virtual const char* getPluginNamespace() const override { return ""; }
    virtual nvinfer1::IPluginV2* clone() const override { return new DCNv2Plugin{_in_channel,_out_channel,_kernel_H,_kernel_W,_deformable_group,_dilation,_groups,_padding,_stride,_weight,_bias}; }

    bool supportsFormat(nvinfer1::DataType type,
                        nvinfer1::PluginFormat format) const override;

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
    ~DCNv2Plugin();
};

class DCNv2PluginCreator : public nvinfer1::IPluginCreator
{
public:
  DCNv2PluginCreator() {}

  ~DCNv2PluginCreator() {}

  const char* getPluginName() const { return DCNV2_PLUGIN_NAME; }

  const char* getPluginVersion() const { return DCNV2_PLUGIN_VERSION; }

  const nvinfer1::PluginFieldCollection* getFieldNames() { std::cerr<< "Function not implemented" << std::endl; return nullptr; }

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) { std::cerr<< "Function not implemented" << std::endl; return nullptr; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) { return new DCNv2Plugin{serialData, serialLength}; }

  void setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }

  const char* getPluginNamespace() const { return mNamespace.c_str(); }
private:
    std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(DCNv2PluginCreator);

#endif //ONNX2TRT_DCNV2_H
