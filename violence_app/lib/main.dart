// lib/main.dart
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'theme/app_theme.dart';
import 'screens/onboarding_screen.dart';
import 'screens/home_screen.dart';
import 'services/sound_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final prefs = await SharedPreferences.getInstance();
  final onboarded = prefs.getBool('onboarded') ?? false;
  await SoundService.instance.init(); // Load sound prefs
  runApp(SafeWatchApp(onboarded: onboarded));
}

class SafeWatchApp extends StatelessWidget {
  final bool onboarded;
  const SafeWatchApp({super.key, required this.onboarded});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SafeWatch — Violence Detection',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.dark,
      home: onboarded ? const HomeScreen() : const OnboardingScreen(),
    );
  }
}
