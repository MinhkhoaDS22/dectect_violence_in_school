import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:violence_app/main.dart';

void main() {
  testWidgets('App starts', (WidgetTester tester) async {
    await tester.pumpWidget(const SafeWatchApp(onboarded: false));
    expect(find.text('SafeWatch'), findsAny);
  });
}
