// ignore_for_file: avoid_print

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:inference_test/utils/ml-performance/common.dart';
import 'package:inference_test/utils/results.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:inference_test/utils/data.dart';
import 'package:inference_test/utils/models.dart';

class ONNXRuntimePerformanceTester extends PerformanceTester {
  @override
  String get libraryName => "onnxruntime";

  @override
  Future<MLInferencePerformanceResult> testPerformance(
      {required LoadModelOptions loadModelOptions}) async {
    print(
        "Loading model: ${loadModelOptions.model} with input precision: ${loadModelOptions.inputPrecision}");

    final session = await _getOrtSessionAsync(loadModelOptions);

    final runOptions = OrtRunOptions();

    // wait for the model to be ready
    await Future.delayed(const Duration(seconds: 1));

    var fastestTimeMs = double.infinity;
    var slowestTimeMs = 0.0;
    var sum = 0.0;

    var resultSender = ResultSender();
    var resultsId = DateTime.now().millisecondsSinceEpoch.toString();
    var sendResultsOptions = SendResultsOptions<dynamic>(
        resultsId: resultsId,
        inputIndex: 0,
        precision: loadModelOptions.inputPrecision,
        library: libraryName,
        output: null,
        inferenceTimeMs: 0,
        model: loadModelOptions.model,
        delegate: loadModelOptions.delegate);

    var batchSize = 25;
    var maxAmount = loadModelOptions.model == Model.deeplabv3 ? 100 : 300;

    print("Running inference");
    for (var i = 0; i < maxAmount;) {
      List<SendResultsOptions<dynamic>> results = [];
      var options = FetchImageDataOptions(
          amount: batchSize,
          dataset: getModelDataSet(loadModelOptions.model),
          skip: i);
      var data = await DataService().fetchImageData(options: options);

      var inputs = await _getFormattedInputs(
          loadModelOptions: loadModelOptions,
          data: data,
          inputNames: session.inputNames);

      for (var data in inputs) {
        print("run");
        var start = DateTime.now();
        final outputs =
            await session.runAsync(runOptions, data, session.outputNames);
        print("Run done");
        var end = DateTime.now();

        var timeMs = end.difference(start).inMilliseconds;

        if (loadModelOptions.model == Model.mobilenet_edgetpu ||
            loadModelOptions.model == Model.mobilenetv2) {
          dynamic o = outputs?[0]?.value;
          var t = o[0];
          sendResultsOptions.output = t;
        }

        if (loadModelOptions.model == Model.ssd_mobilenet) {
          dynamic o0 = outputs?[0]?.value;
          dynamic o1 = outputs?[1]?.value;
          dynamic o2 = outputs?[2]?.value;
          dynamic o3 = outputs?[3]?.value;

          sendResultsOptions.output = [o0[0][0], o1[0], o2[0], o3];
        }

        if (loadModelOptions.model == Model.deeplabv3) {
          dynamic o = outputs?[0]?.value;
          sendResultsOptions.output = o[0];
        }
        sendResultsOptions.inferenceTimeMs = timeMs.toDouble();
        sendResultsOptions.inputIndex = i;
        results.add(sendResultsOptions);
        print(timeMs);

        if (timeMs < fastestTimeMs) {
          fastestTimeMs = timeMs.toDouble();
        }
        if (timeMs > slowestTimeMs) {
          slowestTimeMs = timeMs.toDouble();
        }
        sum += timeMs;

        for (var key in data.keys) {
          data[key]?.release();
        }

        for (var output in outputs ?? []) {
          output?.release();
        }
        i++;
      }

      await resultSender.sendMultipleResultsAsync(
          loadModelOptions.model, results);
    }
    session.release();
    runOptions.release();

    OrtEnv.instance.release();

    return MLInferencePerformanceResult(
        avgPerformanceTimeMs: sum / maxAmount,
        fastestTimeMs: fastestTimeMs,
        slowestTimeMs: slowestTimeMs);
  }

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

  Future<List<Map<String, OrtValue>>> _getInputs(
      {required List<OrtValueTensor> tensors,
      required List<String> inputNames}) async {
    var inputName = inputNames[0];
    var result = tensors.map((e) {
      var input = e;
      return {inputName: input};
    }).toList();

    return result;
  }

  Future<List<Map<String, OrtValue>>> _getFormattedInputs(
      {required LoadModelOptions loadModelOptions,
      required List<FetchImagesQueryData> data,
      required List<String> inputNames}) async {
    var tensors =
        await _getTensors(loadModelOptions: loadModelOptions, data: data);

    return _getInputs(tensors: tensors, inputNames: inputNames);
  }

  Future<List<OrtValueTensor>> _getTensors(
      {required LoadModelOptions loadModelOptions,
      required List<FetchImagesQueryData> data}) async {
    var inputShape = getInputShape(loadModelOptions.model);

    var precisionData = formatImageDataToPrecision(
        data, loadModelOptions.inputPrecision, loadModelOptions.model);

    var tensors = precisionData.map((e) {
      if (loadModelOptions.inputPrecision == InputPrecision.uint8) {
        var tensor = OrtValueTensor.createTensorWithDataList(
            Uint8List.fromList(e as List<int>), [1, ...inputShape]);
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
}
