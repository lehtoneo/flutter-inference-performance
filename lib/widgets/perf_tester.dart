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

  int _run = 0;
  int _runTimes = 5;
  bool _isRunning = false;
  String? _error;
  MLInferencePerformanceResult? _inferencePerformanceResult;
  LoadModelOptions _loadModelOptions = LoadModelOptions(
      model: Model.mobilenet_edgetpu,
      inputPrecision: InputPrecision.uint8,
      delegate: DelegateOption.cpu);

  List<Model> get models => Model.values;
  List<InputPrecision> get precisions => InputPrecision.values;
  List<DelegateOption> get delegates => performanceTester.getDelegateOptions();
  int get testsAmount => models.length * precisions.length * delegates.length;
  List<String> performedRuns = [];
  String? currentRun;

  Future<String?> _runInference(LoadModelOptions loadModelOptions) async {
    if (_isRunning) return null;
    setState(() {
      _inferencePerformanceResult = null;
      _isRunning = true;
      _error = null;
    });

    int run = 0;
    var optionString = loadModelOptions.toString();
    print("Running with options: $optionString");

    while (run < _runTimes) {
      print("Running for the $_run time");
      if (run > 0) {
        print("Waiting for 3 seconds");
        await Future.delayed(Duration(seconds: 3));
      }
      try {
        var result = await performanceTester.testPerformance(
            loadModelOptions: loadModelOptions);

        setState(() {
          _inferencePerformanceResult = result;
        });
      } catch (e) {
        setState(() {
          _error = e.toString();
          _isRunning = false;
        });
        return e.toString();
      }
      run++;
      setState(() {
        _run = run;
      });
    }
    setState(() {
      _isRunning = false;
    });
    return null;
  }

  void _runAll() async {
    for (var model in models) {
      for (var precision in precisions) {
        for (var delegate in delegates) {
          setState(() {
            currentRun = "Running $model with $precision and $delegate";
          });

          print("Sleeping for 3 seconds");
          await Future.delayed(Duration(seconds: 3));

          var loadModelOptions = LoadModelOptions(
            model: model,
            inputPrecision: precision,
            delegate: delegate,
          );
          setState(() {
            _loadModelOptions = loadModelOptions;
          });
          await _runInference(loadModelOptions);

          setState(() {
            performedRuns.add(currentRun!);
          });
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Column(
        children: [
          Text(performanceTester.libraryName, style: TextStyle(fontSize: 24)),
          Text("Tests amount: $testsAmount", style: TextStyle(fontSize: 24)),
          Text("Performed runs: ${performedRuns.length}"),
          Text("Current run $currentRun"),
          ElevatedButton(
            onPressed: _runAll,
            child: const Text("Run all speed tests"),
          ),
          Text("Select options:"),
          LoadModelOptionsSelector(
            initialOptions: _loadModelOptions,
            onOptionsChanged: (newOptions) {
              setState(() {
                _loadModelOptions = newOptions;
              });
            },
            delegates: performanceTester.getDelegateOptions(),
          ),
          Text("Run  $_run / $_runTimes times"),
          ElevatedButton(
            onPressed: () => _runInference(_loadModelOptions),
            child: const Text("Run one speed test"),
          ),
          ElevatedButton(
              onPressed: () => performanceTester.runResourcesTest(
                  loadModelOptions: _loadModelOptions),
              child: const Text("Run resources test")),
          if (_error != null)
            Text(
              "Error: $_error",
              style: TextStyle(color: Colors.red),
            ),
          if (_isRunning) CircularProgressIndicator(),
          InferenceResultDisplay(
              performanceResult: _inferencePerformanceResult),
        ],
      ),
    );
  }
}
