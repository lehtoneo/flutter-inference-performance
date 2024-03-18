import 'package:flutter/material.dart';
import 'package:inference_test/utils/models.dart';
import 'package:inference_test/utils/ml-performance.dart';
import 'package:inference_test/widgets/inference_result_display.dart';
import 'package:inference_test/widgets/load_model_options_selector.dart';

class TfTest extends StatefulWidget {
  const TfTest({Key? key}) : super(key: key);

  @override
  _TfTestState createState() => _TfTestState();
}

class _TfTestState extends State<TfTest> {
  MLInferencePerformanceResult? _inferencePerformanceResult;
  bool _isRunning = false;
  LoadModelOptions _loadModelOptions = LoadModelOptions(
      model: Model.mobilenet, inputPrecision: InputPrecision.float32);

  void _runInference() async {
    if (_isRunning) return;
    setState(() {
      _isRunning = true;
    });
    var tt = TFLitePerformanceTester();

    var result = await tt.testPerformance(loadModelOptions: _loadModelOptions);

    setState(() {
      _isRunning = false;
      _inferencePerformanceResult = result;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text("Model"),
            LoadModelOptionsSelector(
              initialOptions: _loadModelOptions,
              onOptionsChanged: (newOptions) {
                setState(() {
                  _loadModelOptions = newOptions;
                });
              },
            ),
            ElevatedButton(
                onPressed: _runInference, child: const Text("Run Inference")),
            InferenceResultDisplay(
                performanceResult: _inferencePerformanceResult),
          ],
        ),
      ),
    );
  }
}
