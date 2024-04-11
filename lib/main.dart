import 'dart:async';
import 'package:flutter/material.dart';
import 'package:inference_test/pages/home.dart';
import 'package:inference_test/pages/onnx_test.dart';
import 'package:inference_test/pages/tf_test.dart';
import 'package:inference_test/utils/data.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  bool _isErrorCondition = false;
  late Timer _timer;

  @override
  void initState() {
    super.initState();
    _checkCondition();
  }

  Future _checkCondition() async {
    // Implement your condition check logic here
    bool isReachable =
        await DataService.isReachable(); // Result of the condition check

    setState(() {
      _isErrorCondition = !isReachable;
    });
  }

  @override
  void dispose() {
    _timer.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isErrorCondition) {
      return MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: Text("Error")),
          body: Center(
              child: Text("API not reachable. Make sure data API is running")),
        ),
      );
    } else {
      return MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: ThemeData(primarySwatch: Colors.deepPurple),
        home: const HomePage(),
        routes: {
          '/tf_test': (context) => const TfTest(),
          '/home': (context) => const HomePage(),
          '/onnx_test': (context) => const OnnxTest()
        },
      );
    }
  }
}
