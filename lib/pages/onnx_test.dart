import 'package:flutter/material.dart';
import 'package:inference_test/utils/ml-performance/common.dart';
import 'package:inference_test/utils/ml-performance/onnxruntime.dart';
import 'package:inference_test/utils/models.dart';
import 'package:inference_test/widgets/inference_result_display.dart';
import 'package:inference_test/widgets/load_model_options_selector.dart';

class OnnxTest extends StatefulWidget {
  const OnnxTest({Key? key}) : super(key: key);

  @override
  _OnnxTestState createState() => _OnnxTestState();
}

class _OnnxTestState extends State<OnnxTest> {
  bool _isRunning = false;
  MLInferencePerformanceResult? _inferencePerformanceResult;
  LoadModelOptions _loadModelOptions = LoadModelOptions(
      model: Model.mobilenet_edgetpu, inputPrecision: InputPrecision.uint8);

  void _runInference() async {
    if (_isRunning) return;
    setState(() {
      _isRunning = true;
    });
    ONNXRuntimePerformanceTester runner = ONNXRuntimePerformanceTester();
    var result =
        await runner.testPerformance(loadModelOptions: _loadModelOptions);
    setState(() {
      _inferencePerformanceResult = result;
      _isRunning = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            LoadModelOptionsSelector(
              initialOptions: _loadModelOptions,
              onOptionsChanged: (newOptions) {
                setState(() {
                  _loadModelOptions = newOptions;
                });
              },
            ),
            ElevatedButton(onPressed: _runInference, child: const Text("Run")),
            InferenceResultDisplay(
                performanceResult: _inferencePerformanceResult),
          ],
        ),
      ),
    );
  }
}
