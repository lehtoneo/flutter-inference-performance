import 'package:flutter/material.dart';
import 'package:inference_test/utils/models.dart';

class LoadModelOptionsSelector extends StatefulWidget {
  final LoadModelOptions initialOptions;
  final Function(LoadModelOptions) onOptionsChanged;

  const LoadModelOptionsSelector({
    Key? key,
    required this.initialOptions,
    required this.onOptionsChanged,
  }) : super(key: key);

  @override
  // ignore: library_private_types_in_public_api
  _LoadModelOptionsSelectorState createState() =>
      _LoadModelOptionsSelectorState();
}

class _LoadModelOptionsSelectorState extends State<LoadModelOptionsSelector> {
  late Model _selectedModel;
  late InputPrecision _selectedPrecision;

  @override
  void initState() {
    super.initState();
    _selectedModel = widget.initialOptions.model;
    _selectedPrecision = widget.initialOptions.inputPrecision;
  }

  void _updateOptions() {
    widget.onOptionsChanged(
      LoadModelOptions(
          model: _selectedModel, inputPrecision: _selectedPrecision),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        DropdownButton<Model>(
          value: _selectedModel,
          items: Model.values.map((Model value) {
            return DropdownMenuItem<Model>(
              value: value,
              child: Text(value.toString()),
            );
          }).toList(),
          onChanged: (model) {
            setState(() {
              _selectedModel = model!;
            });
            _updateOptions();
          },
        ),
        DropdownButton<InputPrecision>(
          value: _selectedPrecision,
          items: InputPrecision.values.map((InputPrecision value) {
            return DropdownMenuItem<InputPrecision>(
              value: value,
              child: Text(value.toString()),
            );
          }).toList(),
          onChanged: (inputPrecision) {
            setState(() {
              _selectedPrecision = inputPrecision!;
            });
            _updateOptions();
          },
        ),
      ],
    );
  }
}
