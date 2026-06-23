// lib/screens/dashboard_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:fl_chart/fl_chart.dart';
import '../theme/app_theme.dart';
import '../models/alert_model.dart';
import '../services/api_service.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final _api = ApiService();
  List<AlertModel> _alerts = [];
  bool _loading = true;
  String? _error;
  bool _isConnectionError = false;
  String _serverUrl = '';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
      _isConnectionError = false;
    });
    _serverUrl = await ApiService.getBaseUrl();
    try {
      final data = await _api.getAlerts(limit: 200);
      if (mounted) setState(() => _alerts = data);
    } catch (e) {
      if (mounted) {
        final msg = e.toString();
        // Lỗi kết nối → treat as empty (không hiện lỗi kỹ thuật)
        _isConnectionError = msg.contains('connection') ||
            msg.contains('XMLHttpRequest') ||
            msg.contains('SocketException') ||
            msg.contains('Connection refused') ||
            msg.contains('Failed host lookup');
        // Chỉ đặt _error nếu không phải lỗi kết nối
        if (!_isConnectionError) setState(() => _error = msg);
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 720),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Header
              Row(
                children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      color: AppColors.primary.withOpacity(.12),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.bar_chart_rounded,
                        color: AppColors.primary, size: 24),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Dashboard',
                            style: Theme.of(context).textTheme.displayMedium),
                        Text('Thống kê tổng quan hệ thống',
                            style: Theme.of(context).textTheme.bodyLarge),
                      ],
                    ),
                  ),
                  IconButton(
                    onPressed: _load,
                    icon: const Icon(Icons.refresh_rounded),
                    tooltip: 'Làm mới',
                  ),
                ],
              ),
              const SizedBox(height: 24),

              if (_loading)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 60),
                  child: Center(
                      child: CircularProgressIndicator(
                          color: AppColors.primary)),
                )
              // Lỗi kết nối → hiện giao diện hướng dẫn kết nối
              else if (_error != null && _isConnectionError)
                _ConnectionErrorState(serverUrl: _serverUrl, onRetry: _load)
              // Lỗi thực sự (không phải kết nối)
              else if (_error != null)
                _ErrorState(error: _error!, onRetry: _load)
              else
                _DashboardContent(alerts: _alerts),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Dashboard content ─────────────────────────────────────
class _DashboardContent extends StatelessWidget {
  final List<AlertModel> alerts;
  const _DashboardContent({required this.alerts});

  @override
  Widget build(BuildContext context) {
    final total = alerts.length;
    final violence = alerts.where((a) => a.isViolence).length;
    final nonViolence = total - violence;
    final rate = total > 0 ? (violence / total * 100) : 0.0;

    final now = DateTime.now();
    final days = List.generate(7, (i) => now.subtract(Duration(days: 6 - i)));
    final dailyCounts = days.map((d) => alerts
        .where((a) =>
            a.timestamp.year == d.year &&
            a.timestamp.month == d.month &&
            a.timestamp.day == d.day)
        .length).toList();
    final dailyViolence = days.map((d) => alerts
        .where((a) =>
            a.isViolence &&
            a.timestamp.year == d.year &&
            a.timestamp.month == d.month &&
            a.timestamp.day == d.day)
        .length).toList();

    return Column(
      children: [
        // ── KPI cards ──────────────────────────────────────
        Row(
          children: [
            _KpiCard(
              icon: Icons.video_library_rounded,
              label: 'Tổng phân tích',
              value: '$total',
              color: AppColors.primary,
            ),
            const SizedBox(width: 12),
            _KpiCard(
              icon: Icons.warning_amber_rounded,
              label: 'Phát hiện BL',
              value: '$violence',
              color: total > 0 ? AppColors.danger : AppColors.textSecondary,
            ),
            const SizedBox(width: 12),
            _KpiCard(
              icon: Icons.percent_rounded,
              label: 'Tỷ lệ BL',
              value: total > 0 ? '${rate.toStringAsFixed(1)}%' : '—',
              color: rate > 30 ? AppColors.danger : AppColors.warning,
            ),
          ],
        ).animate().fadeIn(duration: 400.ms),
        const SizedBox(height: 20),

        // ── Chưa có dữ liệu ────────────────────────────────
        if (total == 0) ...[
          const _EmptyDataState(),
        ] else ...[
          // ── Donut chart ────────────────────────────────────
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Phân loại kết quả',
                      style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 16),
                  SizedBox(
                    height: 200,
                    child: Row(
                      children: [
                        Expanded(
                          child: PieChart(
                            PieChartData(
                              sectionsSpace: 3,
                              centerSpaceRadius: 50,
                              sections: [
                                PieChartSectionData(
                                  value: violence.toDouble(),
                                  color: AppColors.danger,
                                  title: violence > 0 ? '$violence' : '',
                                  titleStyle: const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.w700,
                                      fontSize: 14),
                                  radius: 60,
                                ),
                                PieChartSectionData(
                                  value: nonViolence.toDouble(),
                                  color: AppColors.success,
                                  title: nonViolence > 0 ? '$nonViolence' : '',
                                  titleStyle: const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.w700,
                                      fontSize: 14),
                                  radius: 60,
                                ),
                              ],
                            ),
                          ),
                        ),
                        const SizedBox(width: 24),
                        Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            _DonutLegend(
                                color: AppColors.danger,
                                label: 'Bạo lực',
                                count: violence),
                            const SizedBox(height: 12),
                            _DonutLegend(
                                color: AppColors.success,
                                label: 'Bình thường',
                                count: nonViolence),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ).animate().fadeIn(delay: 200.ms),
          const SizedBox(height: 16),

          // ── Bar chart 7 ngày ───────────────────────────────
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Hoạt động 7 ngày gần nhất',
                      style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 16),
                  SizedBox(
                    height: 180,
                    child: BarChart(
                      BarChartData(
                        alignment: BarChartAlignment.spaceAround,
                        maxY: (dailyCounts.fold(
                                    0, (a, b) => a > b ? a : b) +
                                2)
                            .toDouble(),
                        barTouchData: BarTouchData(
                          touchTooltipData: BarTouchTooltipData(
                            getTooltipColor: (_) => AppColors.surface,
                            getTooltipItem: (group, _, rod, rodIndex) {
                              final d = days[group.x];
                              return BarTooltipItem(
                                '${d.day}/${d.month}\n'
                                '${rod.toY.toInt()} video',
                                const TextStyle(
                                    color: AppColors.textPrimary,
                                    fontSize: 12),
                              );
                            },
                          ),
                        ),
                        titlesData: FlTitlesData(
                          leftTitles: const AxisTitles(
                              sideTitles: SideTitles(showTitles: false)),
                          topTitles: const AxisTitles(
                              sideTitles: SideTitles(showTitles: false)),
                          rightTitles: const AxisTitles(
                              sideTitles: SideTitles(showTitles: false)),
                          bottomTitles: AxisTitles(
                            sideTitles: SideTitles(
                              showTitles: true,
                              getTitlesWidget: (val, _) {
                                final d = days[val.toInt()];
                                return Text('${d.day}/${d.month}',
                                    style: Theme.of(context)
                                        .textTheme
                                        .bodyMedium
                                        ?.copyWith(fontSize: 9));
                              },
                            ),
                          ),
                        ),
                        gridData: const FlGridData(show: false),
                        borderData: FlBorderData(show: false),
                        barGroups: List.generate(7, (i) {
                          return BarChartGroupData(
                            x: i,
                            barRods: [
                              BarChartRodData(
                                toY: dailyCounts[i].toDouble(),
                                width: 28,
                                borderRadius: const BorderRadius.vertical(
                                    top: Radius.circular(6)),
                                rodStackItems: [
                                  BarChartRodStackItem(
                                      0,
                                      dailyViolence[i].toDouble(),
                                      AppColors.danger.withOpacity(.8)),
                                  BarChartRodStackItem(
                                      dailyViolence[i].toDouble(),
                                      dailyCounts[i].toDouble(),
                                      AppColors.success.withOpacity(.6)),
                                ],
                              ),
                            ],
                          );
                        }),
                      ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      _DonutLegend(
                          color: AppColors.danger,
                          label: 'Bạo lực',
                          count: null),
                      const SizedBox(width: 16),
                      _DonutLegend(
                          color: AppColors.success,
                          label: 'Bình thường',
                          count: null),
                    ],
                  ),
                ],
              ),
            ),
          ).animate().fadeIn(delay: 300.ms),
        ],
      ],
    );
  }
}

// ── KPI Card ───────────────────────────────────────────────
class _KpiCard extends StatelessWidget {
  final IconData icon;
  final String label, value;
  final Color color;
  const _KpiCard(
      {required this.icon,
      required this.label,
      required this.value,
      required this.color});

  @override
  Widget build(BuildContext context) => Expanded(
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: color.withOpacity(.08),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: color.withOpacity(.2)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icon, color: color, size: 22),
              const SizedBox(height: 12),
              Text(value,
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      color: color,
                      fontWeight: FontWeight.w700,
                      fontSize: 24)),
              Text(label,
                  style: Theme.of(context)
                      .textTheme
                      .bodyMedium
                      ?.copyWith(fontSize: 11)),
            ],
          ),
        ),
      );
}

// ── Donut Legend ───────────────────────────────────────────
class _DonutLegend extends StatelessWidget {
  final Color color;
  final String label;
  final int? count;
  const _DonutLegend(
      {required this.color, required this.label, this.count});

  @override
  Widget build(BuildContext context) => Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 12,
            height: 12,
            decoration: BoxDecoration(
                color: color, borderRadius: BorderRadius.circular(3)),
          ),
          const SizedBox(width: 6),
          Text(
            count != null ? '$label ($count)' : label,
            style: Theme.of(context)
                .textTheme
                .bodyMedium
                ?.copyWith(fontSize: 12),
          ),
        ],
      );
}

// ── Empty data state (dùng cho cả chưa có data và lỗi kết nối) ──
class _EmptyDataState extends StatelessWidget {
  final String? serverUrl;
  final VoidCallback? onRetry;
  const _EmptyDataState({this.serverUrl, this.onRetry});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 40),
        child: Column(
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: AppColors.primary.withOpacity(.08),
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Icon(Icons.analytics_outlined,
                  size: 40, color: AppColors.primary),
            ),
            const SizedBox(height: 20),
            Text('Chưa có dữ liệu thống kê',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Text(
              'Phân tích video ở tab Monitor hoặc Hàng chờ\n'
              'để bắt đầu thu thập thống kê.',
              style: Theme.of(context).textTheme.bodyLarge,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ).animate().fadeIn();
}

// ── Connection error state ─────────────────────────────────
class _ConnectionErrorState extends StatelessWidget {
  final String serverUrl;
  final VoidCallback onRetry;
  const _ConnectionErrorState(
      {required this.serverUrl, required this.onRetry});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 24),
        child: Column(
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: AppColors.danger.withOpacity(.08),
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Icon(Icons.wifi_off_rounded,
                  size: 40, color: AppColors.danger),
            ),
            const SizedBox(height: 20),
            Text('Không kết nối được server',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Text('Đang cố kết nối tới:',
                style: Theme.of(context).textTheme.bodyLarge),
            const SizedBox(height: 6),
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: AppColors.surfaceAlt,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: AppColors.divider),
              ),
              child: Text(
                serverUrl,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: AppColors.primary,
                      fontWeight: FontWeight.w600,
                    ),
              ),
            ),
            const SizedBox(height: 16),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppColors.surfaceAlt,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppColors.divider),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('⚙️  Hướng dẫn khắc phục:',
                      style: Theme.of(context)
                          .textTheme
                          .bodyMedium
                          ?.copyWith(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 8),
                  const _Hint('1. Kiểm tra backend FastAPI đã chạy chưa'),
                  const _Hint('2. Vào tab Cài đặt → đổi Backend URL'),
                  const _Hint('3. Đảm bảo app và server cùng mạng'),
                  const _Hint('4. Kiểm tra firewall / CORS của server'),
                ],
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh_rounded, size: 18),
              label: const Text('Thử lại'),
            ),
          ],
        ),
      ).animate().fadeIn();
}

// ── General error state ────────────────────────────────────
class _ErrorState extends StatelessWidget {
  final String error;
  final VoidCallback onRetry;
  const _ErrorState({required this.error, required this.onRetry});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 40),
        child: Center(
          child: Column(
            children: [
              const Icon(Icons.error_outline_rounded,
                  size: 48, color: AppColors.danger),
              const SizedBox(height: 16),
              Text('Không tải được dữ liệu',
                  style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Text(error,
                  textAlign: TextAlign.center,
                  style: Theme.of(context)
                      .textTheme
                      .bodyMedium
                      ?.copyWith(fontSize: 11)),
              const SizedBox(height: 24),
              ElevatedButton.icon(
                onPressed: onRetry,
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('Thử lại'),
              ),
            ],
          ),
        ),
      );
}

// ── Hint row ───────────────────────────────────────────────
class _Hint extends StatelessWidget {
  final String text;
  const _Hint(this.text);

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 4),
        child: Row(
          children: [
            const Icon(Icons.arrow_right_rounded,
                size: 16, color: AppColors.textSecondary),
            const SizedBox(width: 4),
            Expanded(
              child: Text(text,
                  style: Theme.of(context)
                      .textTheme
                      .bodyMedium
                      ?.copyWith(fontSize: 12)),
            ),
          ],
        ),
      );
}
