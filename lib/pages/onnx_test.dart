import 'package:flutter/material.dart';
import 'package:inference_test/utils/ml-performance/onnxruntime.dart';
import 'package:inference_test/widgets/perf_tester.dart';

class OnnxTest extends StatefulWidget {
  const OnnxTest({Key? key}) : super(key: key);

  @override
  _OnnxTestState createState() => _OnnxTestState();
}

class _OnnxTestState extends State<OnnxTest> {
  var runner = ONNXRuntimePerformanceTester();
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: SingleChildScrollView(
            child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
              PerformanceTesterWidget(performanceTester: runner)
            ])),
      ),
    );
  }
}
