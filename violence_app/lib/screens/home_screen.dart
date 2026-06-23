// lib/screens/home_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../theme/app_theme.dart';
import '../services/api_service.dart';
import 'dashboard_screen.dart';
import 'monitor_screen.dart';
import 'queue_screen.dart';
import 'history_screen.dart';
import 'settings_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _navIndex = 0;
  int _unreadCount = 0;   // Badge: số cảnh báo bạo lực chưa đọc

  final _screens = const [
    DashboardScreen(),
    MonitorScreen(),
    QueueScreen(),
    HistoryScreen(),
    SettingsScreen(),
  ];

  @override
  void initState() {
    super.initState();
    _loadBadge();
  }

  Future<void> _loadBadge() async {
    try {
      final api = ApiService();
      final alerts = await api.getAlerts(limit: 100);
      final p = await SharedPreferences.getInstance();
      final lastRead = p.getInt('last_read_alerts_count') ?? 0;
      final violenceCount = alerts.where((a) => a.isViolence).length;
      final unread = (violenceCount - lastRead).clamp(0, 99);
      if (mounted) setState(() => _unreadCount = unread);
    } catch (_) {}
  }

  Future<void> _markHistoryRead() async {
    try {
      final api = ApiService();
      final alerts = await api.getAlerts(limit: 100);
      final violenceCount = alerts.where((a) => a.isViolence).length;
      final p = await SharedPreferences.getInstance();
      await p.setInt('last_read_alerts_count', violenceCount);
      if (mounted) setState(() => _unreadCount = 0);
    } catch (_) {}
  }

  void _onNavChanged(int i) {
    setState(() => _navIndex = i);
    if (i == 3) _markHistoryRead();   // Tab Lịch sử = index 3
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: _SafeWatchAppBar(currentIndex: _navIndex),
      body: AnimatedSwitcher(
        duration: const Duration(milliseconds: 300),
        switchInCurve: Curves.easeOut,
        child: KeyedSubtree(
          key: ValueKey(_navIndex),
          child: _screens[_navIndex],
        ),
      ),
      bottomNavigationBar: _BottomNav(
        current: _navIndex,
        unreadCount: _unreadCount,
        onChange: _onNavChanged,
      ),
    );
  }
}

class _SafeWatchAppBar extends StatelessWidget implements PreferredSizeWidget {
  final int currentIndex;
  const _SafeWatchAppBar({required this.currentIndex});

  static const _titles = [
    'Dashboard',
    'Phân tích',
    'Hàng chờ',
    'Lịch sử',
    'Cài đặt',
  ];

  @override
  Size get preferredSize => const Size.fromHeight(64);

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 64,
      decoration: BoxDecoration(
        color: AppColors.bg,
        border: Border(
          bottom: BorderSide(
            color: AppColors.divider.withOpacity(.5),
            width: .5,
          ),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Row(
          children: [
            // Logo
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(10),
                gradient: const LinearGradient(
                  colors: [Color(0xFF0A84FF), Color(0xFF0055CC)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
              ),
              child: const Icon(Icons.shield_rounded,
                  size: 20, color: Colors.white),
            ),
            const SizedBox(width: 12),
            Text(
              'SafeWatch',
              style: Theme.of(context)
                  .textTheme
                  .titleLarge
                  ?.copyWith(fontSize: 20),
            ),
            const SizedBox(width: 8),
            Text(
              '— ${_titles[currentIndex]}',
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    color: AppColors.textSecondary,
                  ),
            ),
            const Spacer(),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: AppColors.surfaceAlt,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 7,
                    height: 7,
                    decoration: const BoxDecoration(
                      color: AppColors.success,
                      shape: BoxShape.circle,
                    ),
                  )
                      .animate(onPlay: (c) => c.repeat(reverse: true))
                      .fadeOut(duration: 1000.ms),
                  const SizedBox(width: 6),
                  Text('Online',
                      style: Theme.of(context)
                          .textTheme
                          .bodyMedium
                          ?.copyWith(
                            fontSize: 12,
                            color: AppColors.success,
                            fontWeight: FontWeight.w600,
                          )),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _BottomNav extends StatelessWidget {
  final int current;
  final int unreadCount;
  final ValueChanged<int> onChange;
  const _BottomNav(
      {required this.current,
      required this.unreadCount,
      required this.onChange});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        border: Border(
          top: BorderSide(
              color: AppColors.divider.withOpacity(.5), width: .5),
        ),
      ),
      child: NavigationBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        selectedIndex: current,
        onDestinationSelected: onChange,
        height: 64,
        destinations: [
          const NavigationDestination(
            icon: Icon(Icons.bar_chart_outlined),
            selectedIcon: Icon(Icons.bar_chart_rounded),
            label: 'Dashboard',
          ),
          const NavigationDestination(
            icon: Icon(Icons.video_library_outlined),
            selectedIcon: Icon(Icons.video_library_rounded),
            label: 'Monitor',
          ),
          const NavigationDestination(
            icon: Icon(Icons.queue_outlined),
            selectedIcon: Icon(Icons.queue_rounded),
            label: 'Hàng chờ',
          ),
          // History with badge
          NavigationDestination(
            icon: Badge(
              isLabelVisible: unreadCount > 0,
              label: Text(
                unreadCount > 9 ? '9+' : '$unreadCount',
                style: const TextStyle(fontSize: 9),
              ),
              backgroundColor: AppColors.danger,
              child: const Icon(Icons.history_outlined),
            ),
            selectedIcon: Badge(
              isLabelVisible: unreadCount > 0,
              label: Text(
                unreadCount > 9 ? '9+' : '$unreadCount',
                style: const TextStyle(fontSize: 9),
              ),
              backgroundColor: AppColors.danger,
              child: const Icon(Icons.history_rounded),
            ),
            label: 'Lịch sử',
          ),
          const NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings_rounded),
            label: 'Cài đặt',
          ),
        ],
      ),
    );
  }
}
