// ignore_for_file: avoid_print

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:inference_test/utils/ml-performance/common.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:inference_test/utils/data.dart';
import 'package:inference_test/utils/models.dart';

class ONNXRuntimePerformanceTester
    extends PerformanceTester<OrtSession, Map<String, OrtValue>, Null> {
  @override
  String get libraryName => "onnxruntime_flutter";

  @override
  List<DelegateOption> getLibraryDelegateOptions() {
    return [
      DelegateOption.core_ml,
      DelegateOption.nnapi,
      DelegateOption.xnnpack,
      DelegateOption.cpu
    ];
  }

  Future<OrtSession> _getOrtSessionAsync(
      LoadModelOptions loadModelOptions) async {
    var assetFileName = ModelsUtil().getModelPath(loadModelOptions.model,
        loadModelOptions.inputPrecision, ModelFormat.onnx);
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();

    OrtEnv.instance.init();
    final sessionOptions = _getOrtSessionOptions(loadModelOptions);

    final session = OrtSession.fromBuffer(bytes, sessionOptions);
    return session;
  }

  OrtSessionOptions _getOrtSessionOptions(LoadModelOptions loadModelOptions) {
    var sessionOptions = OrtSessionOptions();

    switch (loadModelOptions.delegate) {
      case DelegateOption.core_ml:
        sessionOptions.appendCoreMLProvider(CoreMLFlags.useNone);
        break;
      case DelegateOption.nnapi:
        sessionOptions.appendNnapiProvider(NnapiFlags.useNone);
        break;
      case DelegateOption.xnnpack:
        sessionOptions.appendXnnpackProvider();
        break;
      case DelegateOption.cpu:
        sessionOptions.appendCPUProvider(CPUFlags.useNone);
        break;
      default:
        throw Exception("Unknown delegate");
    }
    return sessionOptions;
  }

  List<Map<String, OrtValue>> _getInputs(
      {required List<OrtValueTensor> tensors,
      required List<String> inputNames}) {
    var inputName = inputNames[0];
    var result = tensors.map((e) {
      var input = e;
      return {inputName: input};
    }).toList();

    return result;
  }

  List<Map<String, OrtValue>> _getFormattedInputs(
      {required LoadModelOptions loadModelOptions,
      required List<FetchImagesQueryData> data,
      required List<String> inputNames}) {
    var tensors = _getTensors(loadModelOptions: loadModelOptions, data: data);

    return _getInputs(tensors: tensors, inputNames: inputNames);
  }

  List<OrtValueTensor> _getTensors(
      {required LoadModelOptions loadModelOptions,
      required List<FetchImagesQueryData> data}) {
    var inputShape = getInputShape(loadModelOptions.model);

    var precisionData = formatImageDataToPrecision(
        data, loadModelOptions.inputPrecision, loadModelOptions.model);

    var tensors = precisionData.map((e) {
      if (loadModelOptions.inputPrecision == InputPrecision.uint8) {
        var tensor = OrtValueTensor.createTensorWithDataList(
            Uint8List.fromList(e as List<int>), [...inputShape]);
        return tensor;
      } else {
        var tensor = OrtValueTensor.createTensorWithDataList(
            Float32List.fromList(e as List<double>), [1, ...inputShape]);
        return tensor;
      }
    }).toList();
    print("Data ready");
    return tensors;
  }

  @override
  Future<void> closeModel(model) {
    model.release();
    OrtEnv.instance.release();
    return Future.value();
  }

  @override
  formatMobileNetEdgeTpuOutput(out) {
    dynamic o = out?[0]?.value;
    var t = o[0];
    return t;
  }

  @override
  Future<List<Map<String, OrtValue>>> getFormattedInputs(
      {required LoadModelOptions loadModelOptions,
      required OrtSession model,
      required int skip,
      required int batchSize}) async {
    var options = FetchImageDataOptions(
        amount: batchSize,
        dataset: getModelDataSet(loadModelOptions.model),
        skip: skip);
    var data = await DataService().fetchImageData(options: options);

    var inputs = _getFormattedInputs(
        loadModelOptions: loadModelOptions,
        data: data,
        inputNames: model.inputNames);

    return inputs;
  }

  @override
  Future<OrtSession> loadModelAsync(LoadModelOptions loadModelOptions) async {
    return await _getOrtSessionAsync(loadModelOptions);
  }

  @override
  Future runInference(
      {required model,
      required input,
      required outputTensor,
      required LoadModelOptions loadModelOptions}) async {
    return model.run(OrtRunOptions(), input, model.outputNames);
  }

  @override
  formatDeepLabV3Output(outputs) {
    dynamic o = outputs?[0]?.value;

    return o[0];
  }

  @override
  formatMobileNetV2Output(out) {
    return formatMobileNetEdgeTpuOutput(out);
  }

  @override
  formatSSDMobileNetOutput(outputs) {
    dynamic o0 = outputs?[0]?.value;
    dynamic o1 = outputs?[1]?.value;
    dynamic o2 = outputs?[2]?.value;
    dynamic o3 = outputs?[3]?.value;

    return [o0[0][0], o1[0], o2[0], o3];
  }

  @override
  Future onAfterInputRun(
      {required Map<String, OrtValue> input, required dynamic output}) {
    for (var key in input.keys) {
      input[key]?.release();
    }

    for (var out in output ?? []) {
      out?.release();
    }

    return Future.value();
  }

  @override
  Null getOutputTensor({required LoadModelOptions loadModelOptions}) {
    // TODO: implement getOutputTensor
    return null;
  }
}
