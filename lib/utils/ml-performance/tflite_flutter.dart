// ignore_for_file: avoid_print

import 'dart:typed_data';

import 'package:inference_test/utils/ml-performance/common.dart';
import 'package:inference_test/utils/results.dart';
import 'package:inference_test/utils/data.dart';
import 'package:inference_test/utils/models.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

enum TFLiteDelegate { coreML, nnapi, gpu, cpu }

class TFLitePerformanceTester extends PerformanceTester {
  @override
  String get libraryName => "tflite_flutter";

  dynamic _inputs;
  LoadModelOptions? _prevLoadModelOptions;

  @override
  List<DelegateOption> getLibraryDelegateOptions() {
    return DelegateOption.values;
  }

  @override
  Future<MLInferencePerformanceResult> testPerformance({
    required LoadModelOptions loadModelOptions,
  }) async {
    var interpreter = await _loadTFLiteModel(loadModelOptions);

    // cache inputs if the same model is used
    final inputs = _prevLoadModelOptions == loadModelOptions
        ? _inputs
        : await _getFormattedImageData(
            model: loadModelOptions.model,
            inputPrecision: loadModelOptions.inputPrecision);

    _prevLoadModelOptions = loadModelOptions;
    _inputs = inputs;

    var output = _getOutPutTensor(
        loadModelOptions.model, loadModelOptions.inputPrecision);

    var fastestTimeMs = double.infinity;
    var slowestTimeMs = 0.0;
    var sum = 0.0;

    ResultSender resultSender = ResultSender();

    var i = 0;
    var resultsId = DateTime.now().millisecondsSinceEpoch.toString();
    List<SendResultsOptions<dynamic>> results = [];
    for (var input in inputs) {
      var startTime = DateTime.now();
      await interpreter.runForMultipleInputs([
        [input]
      ], output);
      var endTime = DateTime.now();

      var timeMs = endTime.difference(startTime).inMilliseconds;

      if (loadModelOptions.model == Model.mobilenet_edgetpu ||
          loadModelOptions.model == Model.mobilenetv2) {
        results.add(SendResultsOptions<List<num>>(
            resultsId: resultsId,
            inputIndex: i,
            precision: loadModelOptions.inputPrecision,
            library: libraryName,
            output: output[0][0],
            inferenceTimeMs: timeMs.toDouble(),
            model: loadModelOptions.model,
            delegate: loadModelOptions.delegate));
      }

      if (loadModelOptions.model == Model.ssd_mobilenet) {
        results.add(SendResultsOptions<List<dynamic>>(
            resultsId: resultsId,
            inputIndex: i,
            precision: loadModelOptions.inputPrecision,
            library: libraryName,
            output: [output[0][0][0], output[1][0], output[2][0], output[3]],
            inferenceTimeMs: timeMs.toDouble(),
            model: loadModelOptions.model,
            delegate: loadModelOptions.delegate));
      }

      if (loadModelOptions.model == Model.deeplabv3) {
        results.add(SendResultsOptions<List<num>>(
            resultsId: resultsId,
            inputIndex: i,
            precision: loadModelOptions.inputPrecision,
            library: libraryName,
            output: output[0][0],
            inferenceTimeMs: timeMs.toDouble(),
            model: loadModelOptions.model,
            delegate: loadModelOptions.delegate));
      }

      if (timeMs < fastestTimeMs) {
        fastestTimeMs = timeMs.toDouble();
      }

      if (timeMs > slowestTimeMs) {
        slowestTimeMs = timeMs.toDouble();
      }

      sum += timeMs;
      i++;

      print("Inference Time: ${timeMs}ms");
    }
    interpreter.close();

    await resultSender.sendMultipleResultsAsync(
        loadModelOptions.model, results);

    return MLInferencePerformanceResult(
        avgPerformanceTimeMs: sum / inputs.length,
        fastestTimeMs: fastestTimeMs,
        slowestTimeMs: slowestTimeMs);
  }

  dynamic _getOutPutTensor(Model model, InputPrecision inputPrecision) {
    switch (model) {
      case Model.mobilenet_edgetpu:
        return {
          0: List.filled(1 * 1001, 0).reshape([1, 1001])
        };
      case Model.mobilenetv2:
        return {
          0: List.filled(1 * 1000, 0).reshape([1, 1000])
        };
      case Model.ssd_mobilenet:
        var s = inputPrecision == InputPrecision.uint8 ? 20 : 10;
        return {
          0: List.filled(1 * s * 4, 0).reshape([1, s, 4]),
          1: List.filled(1 * s, 0).reshape([1, s]),
          2: List.filled(1 * s, 0).reshape([1, s]),
          3: List.filled(1, 0).reshape([1])
        };
      case Model.deeplabv3:
        return {
          0: List.filled(1 * 512 * 512, 0).reshape([1, 512, 512])
        };
      default:
        throw Exception("Unknown model");
    }
  }

  Future<List<dynamic>> _getFormattedImageData({
    required Model model,
    required InputPrecision inputPrecision,
  }) async {
    print("Fetching data for model: ${model.name}");
    var options =
        FetchImageDataOptions(amount: 20, dataset: getModelDataSet(model));

    var data = await DataService().fetchImageData(options: options);
    print("Data fetched");

    var inputShape = getInputShape(model);

    var precisionData = formatImageDataToPrecision(data, inputPrecision, model);
    var reshaped = precisionData.map((e) => e.reshape(inputShape)).toList();

    print("Data formatted");
    return reshaped;
  }

  Future<IsolateInterpreter> _loadTFLiteModel(LoadModelOptions options) async {
    print(
        "Loading model: ${options.model} with input precision: ${options.inputPrecision}");

    var interPreterOptions = _getInterpreterOptions(options);

    var modelPath = ModelsUtil().getModelPath(
        options.model, options.inputPrecision, ModelFormat.tflite);

    final interpreter =
        await Interpreter.fromAsset(modelPath, options: interPreterOptions);

    final isolateInterpreter =
        await IsolateInterpreter.create(address: interpreter.address);

    // lets sleep so that the interpreter is ready to perform inference
    // If this is not done, a error is thrown if inference is performed immediately
    await Future.delayed(const Duration(seconds: 1));

    return isolateInterpreter;
  }

  InterpreterOptions _getInterpreterOptions(LoadModelOptions loadModelOptions) {
    var interPreterOptions = InterpreterOptions();

    // needed delegate, coreML, nnapi or gpu // ..addDelegate(CoreMlDelegate())
    if (loadModelOptions.delegate == DelegateOption.core_ml) {
      interPreterOptions.addDelegate(CoreMlDelegate());
    } else if (loadModelOptions.delegate == DelegateOption.nnapi) {
      interPreterOptions.useNnApiForAndroid = true;
    } else if (loadModelOptions.delegate == DelegateOption.metal) {
      interPreterOptions.useMetalDelegateForIOS = true;
    } else if (loadModelOptions.delegate == DelegateOption.gpu) {
      interPreterOptions.addDelegate(GpuDelegate());
    } else if (loadModelOptions.delegate == DelegateOption.xnnpack) {
      interPreterOptions.addDelegate(XNNPackDelegate());
    } else if (loadModelOptions.delegate == DelegateOption.cpu) {
    } else {
      throw Exception("Unknown delegate");
    }

    return interPreterOptions;
  }
}
