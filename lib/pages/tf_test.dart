import 'package:flutter/material.dart';
import 'package:inference_test/utils/ml-performance/tflite_flutter.dart';
import 'package:inference_test/widgets/perf_tester.dart';

class TfTest extends StatefulWidget {
  const TfTest({Key? key}) : super(key: key);

  @override
  _TfTestState createState() => _TfTestState();
}

class _TfTestState extends State<TfTest> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            PerformanceTesterWidget(
                performanceTester: TFLitePerformanceTester())
          ],
        ),
      ),
    );
  }
}
