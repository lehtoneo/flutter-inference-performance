import 'dart:io';
import 'dart:async';

import 'package:inference_test/utils/common.dart';
import 'package:inference_test/utils/data.dart';
import 'package:inference_test/utils/models.dart';
import 'package:inference_test/utils/results.dart';

class MLInferencePerformanceResult {
  final double avgPerformanceTimeMs;
  final double fastestTimeMs;
  final double slowestTimeMs;

  MLInferencePerformanceResult(
      {required this.avgPerformanceTimeMs,
      required this.fastestTimeMs,
      required this.slowestTimeMs});
}

class PerformanceTesterCommon {
  List<DelegateOption> _getCurrentPlatformDelegateOptions() {
    if (Platform.isAndroid) {
      return [
        DelegateOption.nnapi,
        DelegateOption.gpu,
        DelegateOption.cpu,
        DelegateOption.xnnpack
      ];
    } else if (Platform.isIOS) {
      // ios
      return [
        DelegateOption.core_ml,
        DelegateOption.gpu,
        DelegateOption.metal,
        DelegateOption.cpu,
        DelegateOption.xnnpack
      ];
    } else {
      throw Exception("Unknown platform");
    }
  }

  DataSet getModelDataSet(Model model) {
    var mobileNetDataSet = DataSet.imagenet;
    switch (model) {
      case Model.mobilenet_edgetpu:
        return mobileNetDataSet;
      case Model.mobilenetv2:
        return mobileNetDataSet;
      case Model.ssd_mobilenet:
        return DataSet.coco;
      case Model.deeplabv3:
        return DataSet.ade20k;
      default:
        throw Exception("Unknown model");
    }
  }

  List<int> getInputShape(Model model) {
    var mobileNetInputShape = [1, 224, 224, 3];
    switch (model) {
      case Model.mobilenet_edgetpu:
        return mobileNetInputShape;
      case Model.mobilenetv2:
        return mobileNetInputShape;
      case Model.ssd_mobilenet:
        return [1, 300, 300, 3];
      case Model.deeplabv3:
        return [1, 512, 512, 3];
      default:
        throw Exception("Unknown model");
    }
  }

  List<List<num>> formatImageDataToPrecision(List<FetchImagesQueryData> data,
      InputPrecision inputPrecision, Model model) {
    var asUint8 = data.map((e) => e.rawImageBuffer.data).toList();
    if (inputPrecision == InputPrecision.uint8) {
      return asUint8;
    } else {
      switch (model) {
        case Model.mobilenetv2:
          return asUint8.map((e) {
            return e.map((e) => e.toDouble() / 127.5 - 1).toList();
          }).toList();
        case Model.mobilenet_edgetpu:
          var asDoubles = asUint8.map((e) {
            return e.map((e) => e.toDouble() / 127.5 - 1).toList();
          }).toList();
          return asDoubles;
        case Model.ssd_mobilenet:
          return asUint8.map((e) {
            return e.map((e) => e.toDouble() / 127.5 - 1).toList();
          }).toList();
        case Model.deeplabv3:
          return asUint8.map((e) {
            return e.map((e) => e.toDouble() / 255).toList();
          }).toList();
        default:
          throw Exception("Unknown model");
      }
    }
  }
}

abstract class PerformanceTester<ModelType, InputType, OutputTensorType>
    extends PerformanceTesterCommon {
  String get libraryName;

  Future<MLInferencePerformanceResult> testPerformance({
    required LoadModelOptions loadModelOptions,
  }) async {
    ResultSender resultSender = ResultSender();
    var resultsId = DateTime.now().millisecondsSinceEpoch.toString();
    var hasResults = await resultSender.hasResults(SendResultsApiOptions(
        options: new SendResultsOptions<dynamic>(
            resultsId: resultsId,
            inputIndex: 0,
            precision: loadModelOptions.inputPrecision,
            library: libraryName,
            output: null,
            inferenceTimeMs: 0,
            model: loadModelOptions.model,
            delegate: loadModelOptions.delegate)));

    if (hasResults) {
      throw Exception("Results already exist for this model");
    }

    await testInferenceWorks(loadModelOptions: loadModelOptions);
    var model = await loadModelAsync(loadModelOptions);

    var fastestTimeMs = double.infinity;
    var slowestTimeMs = 0.0;
    var sum = 0.0;

    var batchSize = 10;
    var maxAmount = loadModelOptions.model == Model.deeplabv3 ? 100 : 300;

    for (var i = 0; i < maxAmount;) {
      List<SendResultsOptions<dynamic>> results = [];
      final inputs = await getFormattedInputs(
          loadModelOptions: loadModelOptions,
          model: model,
          skip: i,
          batchSize: batchSize);
      for (var input in inputs) {
        var outputTensor = getOutputTensor(loadModelOptions: loadModelOptions);

        var startTime = DateTime.now();
        var output = await runInference(
            model: model,
            input: input,
            outputTensor: outputTensor,
            loadModelOptions: loadModelOptions);
        var endTime = DateTime.now();

        var timeMs = endTime.difference(startTime).inMilliseconds;

        var sendResultsOptions = new SendResultsOptions<dynamic>(
            resultsId: resultsId,
            inputIndex: i,
            precision: loadModelOptions.inputPrecision,
            library: libraryName,
            output: null,
            inferenceTimeMs: timeMs.toDouble(),
            model: loadModelOptions.model,
            delegate: loadModelOptions.delegate);

        if (loadModelOptions.model == Model.mobilenet_edgetpu) {
          sendResultsOptions.output = formatMobileNetEdgeTpuOutput(output);
        }

        if (loadModelOptions.model == Model.mobilenetv2) {
          sendResultsOptions.output = formatMobileNetV2Output(output);
        }

        if (loadModelOptions.model == Model.ssd_mobilenet) {
          sendResultsOptions.output = formatSSDMobileNetOutput(output);
        }

        if (loadModelOptions.model == Model.deeplabv3) {
          sendResultsOptions.output = formatDeepLabV3Output(output);
        }

        results.add(sendResultsOptions);

        if (timeMs < fastestTimeMs) {
          fastestTimeMs = timeMs.toDouble();
        }

        if (timeMs > slowestTimeMs) {
          slowestTimeMs = timeMs.toDouble();
        }

        await onAfterInputRun(input: input, output: output);

        sum += timeMs;
        i++;
      }

      await resultSender.sendMultipleResultsAsync(
          loadModelOptions.model, results);
      results = [];
    }

    closeModel(model);

    return MLInferencePerformanceResult(
        avgPerformanceTimeMs: sum / maxAmount,
        fastestTimeMs: fastestTimeMs,
        slowestTimeMs: slowestTimeMs);
  }

  ///
  /// - This method is used to test if the inference works
  /// - It is used to test if the model can be loaded and run
  /// - If the model is not run within 10 seconds, it will throw a timeout exception
  Future<void> testInferenceWorks({
    required LoadModelOptions loadModelOptions,
  }) async {
    var model = await loadModelAsync(loadModelOptions);

    var inputs = await getFormattedInputs(
        loadModelOptions: loadModelOptions,
        model: model,
        skip: 0,
        batchSize: 1);

    var input = inputs.first;

    var outputTensor = getOutputTensor(loadModelOptions: loadModelOptions);

    try {
      var output = await _runInferenceWithTimeout(
          model: model,
          input: input,
          outputTensor: outputTensor,
          loadModelOptions: loadModelOptions);

      await onAfterInputRun(input: input, output: output);
    } catch (e) {
      print("Inference failed: $e");
      rethrow;
    }

    closeModel(model);
  }

  Future<dynamic> _runInferenceWithTimeout({
    required ModelType model,
    required dynamic input,
    required dynamic outputTensor,
    required LoadModelOptions loadModelOptions,
  }) async {
    final future = runInference(
      model: model,
      input: input,
      outputTensor: outputTensor,
      loadModelOptions: loadModelOptions,
    );

    return timeout(future, Duration(seconds: 10),
        message: 'Inference timed out');
  }

  List<DelegateOption> getLibraryDelegateOptions();

  Future<ModelType> loadModelAsync(LoadModelOptions loadModelOptions);

  Future<List<InputType>> getFormattedInputs({
    required LoadModelOptions loadModelOptions,
    required ModelType model,
    required int skip,
    required int batchSize,
  });

  dynamic formatMobileNetV2Output(dynamic out);
  dynamic formatMobileNetEdgeTpuOutput(dynamic out);

  dynamic formatSSDMobileNetOutput(dynamic out);

  dynamic formatDeepLabV3Output(dynamic out);

  Future<dynamic> onAfterInputRun(
      {required InputType input, required dynamic output});

  Future<dynamic> runInference(
      {required ModelType model,
      required InputType input,
      required OutputTensorType outputTensor,
      required LoadModelOptions loadModelOptions});

  OutputTensorType getOutputTensor(
      {required LoadModelOptions loadModelOptions});

  Future<void> closeModel(ModelType model);

  List<DelegateOption> getDelegateOptions() {
    var platformDelegateOptions = _getCurrentPlatformDelegateOptions();
    var libraryDelegateOptions = getLibraryDelegateOptions();
    return platformDelegateOptions
        .where((element) => libraryDelegateOptions.contains(element))
        .toList();
  }
}
