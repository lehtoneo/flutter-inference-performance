import 'package:flutter/material.dart';
import 'package:inference_test/utils/ml-performance/common.dart';
import 'package:inference_test/utils/ml-performance/onnxruntime.dart';
import 'package:inference_test/utils/models.dart';
import 'package:inference_test/widgets/inference_result_display.dart';
import 'package:inference_test/widgets/load_model_options_selector.dart';

class PerformanceTesterWidget extends StatefulWidget {
  final PerformanceTester performanceTester;
  const PerformanceTesterWidget({Key? key, required this.performanceTester})
      : super(key: key);

  @override
  _PerformanceTesterWidgetState createState() =>
      _PerformanceTesterWidgetState(performanceTester: performanceTester);
}

class _PerformanceTesterWidgetState extends State<PerformanceTesterWidget> {
  final PerformanceTester performanceTester;
  _PerformanceTesterWidgetState({required this.performanceTester});

  bool _isRunning = false;
  String? _error;
  MLInferencePerformanceResult? _inferencePerformanceResult;
  LoadModelOptions _loadModelOptions = LoadModelOptions(
      model: Model.mobilenet_edgetpu,
      inputPrecision: InputPrecision.uint8,
      delegate: DelegateOption.cpu);

  void _runInference() async {
    if (_isRunning) return;
    setState(() {
      _inferencePerformanceResult = null;
      _isRunning = true;
      _error = null;
    });

    try {
      var result = await performanceTester.testPerformance(
          loadModelOptions: _loadModelOptions);
      setState(() {
        _inferencePerformanceResult = result;
        _isRunning = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isRunning = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        LoadModelOptionsSelector(
          initialOptions: _loadModelOptions,
          onOptionsChanged: (newOptions) {
            setState(() {
              _loadModelOptions = newOptions;
            });
          },
        ),
        ElevatedButton(
          onPressed: _runInference,
          child: const Text("Run"),
        ),
        if (_error != null)
          Text(
            "Error: $_error",
            style: TextStyle(color: Colors.red),
          ),
        if (_isRunning) CircularProgressIndicator(),
        InferenceResultDisplay(performanceResult: _inferencePerformanceResult),
      ],
    );
  }
}
