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

    var ttt = await _getInputs(
        loadModelOptions: loadModelOptions, inputNames: session.inputNames);

    // wait for the model to be ready
    await Future.delayed(const Duration(seconds: 1));

    var fastestTimeMs = double.infinity;
    var slowestTimeMs = 0.0;
    var sum = 0.0;

    print("Running inference");
    var resultSender = ResultSender();
    var i = 0;
    var resultsId = DateTime.now().millisecondsSinceEpoch.toString();

    List<SendResultsOptions<dynamic>> results = [];

    for (var data in ttt) {
      var inputOrt = data;
      print("run");
      var start = DateTime.now();
      final outputs =
          await session.runAsync(runOptions, inputOrt, session.outputNames);
      print("Run done");
      var end = DateTime.now();

      var timeMs = end.difference(start).inMilliseconds;

      if (loadModelOptions.model == Model.mobilenet_edgetpu ||
          loadModelOptions.model == Model.mobilenetv2) {
        dynamic o = outputs?[0]?.value;
        var t = o[0];
        results.add(SendResultsOptions<List<num>>(
            resultsId: resultsId,
            inputIndex: i,
            precision: loadModelOptions.inputPrecision,
            library: libraryName,
            output: t,
            inferenceTimeMs: timeMs.toDouble(),
            model: loadModelOptions.model,
            delegate: loadModelOptions.delegate));
      }

      if (loadModelOptions.model == Model.ssd_mobilenet) {
        dynamic o0 = outputs?[0]?.value;
        dynamic o1 = outputs?[1]?.value;
        dynamic o2 = outputs?[2]?.value;
        dynamic o3 = outputs?[3]?.value;

        results.add(SendResultsOptions<dynamic>(
            resultsId: resultsId,
            inputIndex: i,
            precision: loadModelOptions.inputPrecision,
            library: libraryName,
            output: [o0[0][0], o1[0], o2[0], o3],
            inferenceTimeMs: timeMs.toDouble(),
            model: loadModelOptions.model,
            delegate: loadModelOptions.delegate));
      }

      if (loadModelOptions.model == Model.deeplabv3) {
        print(outputs?[0]?.value);
        dynamic o = outputs?[0]?.value;
        results.add(SendResultsOptions<dynamic>(
            resultsId: resultsId,
            inputIndex: i,
            precision: loadModelOptions.inputPrecision,
            library: libraryName,
            output: o[0],
            inferenceTimeMs: timeMs.toDouble(),
            model: loadModelOptions.model,
            delegate: loadModelOptions.delegate));
      }

      print(timeMs);
      if (timeMs < fastestTimeMs) {
        fastestTimeMs = timeMs.toDouble();
      }
      if (timeMs > slowestTimeMs) {
        slowestTimeMs = timeMs.toDouble();
      }
      sum += timeMs;

      for (var key in inputOrt.keys) {
        inputOrt[key]?.release();
      }

      for (var output in outputs ?? []) {
        output?.release();
      }
      i++;

      if (i % 10 == 0) {
        // send results in batches of 10
        await resultSender.sendMultipleResultsAsync(
            loadModelOptions.model, results);
        results = [];
      }
    }

    print("Inference ready");
    session.release();
    runOptions.release();

    OrtEnv.instance.release();

    return MLInferencePerformanceResult(
        avgPerformanceTimeMs: sum / ttt.length,
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
      {required LoadModelOptions loadModelOptions,
      required List<String> inputNames}) async {
    var tensors = await _getTensors(loadModelOptions: loadModelOptions);

    var inputName = inputNames[0];
    var result = tensors.map((e) {
      var input = e;
      return {inputName: input};
    }).toList();

    // _prevInputs = result;
    //  _prevLoadModelOptions = loadModelOptions;
    return result;
  }

  Future<List<OrtValueTensor>> _getTensors(
      {required LoadModelOptions loadModelOptions}) async {
    var amount = loadModelOptions.model == Model.deeplabv3 ? 100 : 300;
    var options = FetchImageDataOptions(
        amount: amount, dataset: getModelDataSet(loadModelOptions.model));
    var data = await DataService().fetchImageData(options: options);

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
