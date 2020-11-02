//
// Created by yoloCao on 20-5-11.
//

#ifndef TENSORRT_YOLOV4_YOLO_H
#define TENSORRT_YOLOV4_YOLO_H
#pragma once
#include "plugin.hpp"
#include "serialize.hpp"
#include <vector>
#include <cuda_runtime.h>

namespace {
    constexpr const char* YOLO_PLUGIN_VERSION{"001"};
    constexpr const char* YOLO_PLUGIN_NAME{"YOLO"};
}

class YOLOPlugin final : public onnx2trt::PluginV2 {

    bool _initialized;
    std::vector<int> anchors;
    int* cudaAnchors;
    int classes;
    int anchorNum;
    int downStride;
    float inferThresh;
protected:
    void deserialize(void const* serialData, size_t serialLength) {
        deserializeBase(serialData, serialLength);
        deserialize_value(&serialData, &serialLength, &inferThresh);
        deserialize_value(&serialData, &serialLength, &classes);
        deserialize_value(&serialData, &serialLength, &anchorNum);
        deserialize_value(&serialData, &serialLength, &downStride);
        deserialize_value(&serialData, &serialLength, &anchors);
    }
    size_t getSerializationSize() const override {
        return  serialized_size(inferThresh) +serialized_size(classes)+
        serialized_size(anchorNum)+serialized_size(downStride)+serialized_size(anchors)+ getBaseSerializationSize();
    }
    void serialize(void *buffer) const override {
        serializeBase(buffer);
        serialize_value(&buffer, inferThresh);
        serialize_value(&buffer, classes);
        serialize_value(&buffer, anchorNum);
        serialize_value(&buffer, downStride);
        serialize_value(&buffer, anchors);
    }
public:
    YOLOPlugin(const std::vector<int>& anchors, const int anchorNum,
               const int calsses, const int downStride, const float inferThresh);

    YOLOPlugin(void const* serialData, size_t serialLength) : _initialized(false) {
        this->deserialize(serialData, serialLength);
    }
    bool supportsFormat(nvinfer1::DataType type,
                        nvinfer1::PluginFormat format) const override;

    const char* getPluginType() const override { return YOLO_PLUGIN_NAME; }
    virtual const char* getPluginVersion() const override { return YOLO_PLUGIN_VERSION; }
    virtual void setPluginNamespace(const char* pluginNamespace) override {}
    virtual const char* getPluginNamespace() const override { return ""; }

    virtual nvinfer1::IPluginV2* clone() const override { return new YOLOPlugin{anchors,anchorNum,classes,downStride,inferThresh}; }

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
    ~YOLOPlugin();
};

class YOLOPluginCreator : public nvinfer1::IPluginCreator
{
public:
  YOLOPluginCreator() {}

  ~YOLOPluginCreator() {}

  const char* getPluginName() const { return YOLO_PLUGIN_NAME; }

  const char* getPluginVersion() const { return YOLO_PLUGIN_VERSION; }

  const nvinfer1::PluginFieldCollection* getFieldNames() { std::cerr<< "Function not implemented" << std::endl; return nullptr; }

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) { std::cerr<< "Function not implemented" << std::endl; return nullptr; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) { return new YOLOPlugin{serialData, serialLength}; }

  void setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }

  const char* getPluginNamespace() const { return mNamespace.c_str(); }
private:
    std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(YOLOPluginCreator);
#endif //TENSORRT_YOLOV4_YOLO_H
