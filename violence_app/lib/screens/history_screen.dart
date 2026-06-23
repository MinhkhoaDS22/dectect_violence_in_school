import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:intl/intl.dart';
import '../theme/app_theme.dart';
import '../models/alert_model.dart';
import '../services/api_service.dart';
import '../widgets/violence_timeline.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final _api = ApiService();
  List<AlertModel> _alerts = [];
  bool _loading = true;
  String? _error;
  bool _isConnectionError = false;

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
    try {
      final data = await _api.getAlerts();
      if (mounted) setState(() => _alerts = data);
    } catch (e) {
      if (mounted) {
        final msg = e.toString();
        final isConn = msg.contains('connection') ||
            msg.contains('XMLHttpRequest') ||
            msg.contains('SocketException') ||
            msg.contains('Connection refused') ||
            msg.contains('Failed host lookup');
        setState(() {
          _isConnectionError = isConn;
          // Lỗi kết nối → không đặt _error (hiện như empty)
          _error = isConn ? null : msg;
        });
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _delete(AlertModel alert) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: AppColors.surface,
        title: const Text('Xoá cảnh báo?'),
        content: Text('Xoá cảnh báo "${alert.videoFilename}"?'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Huỷ')),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Xoá', style: TextStyle(color: AppColors.danger)),
          ),
        ],
      ),
    );
    if (ok != true) return;
    await _api.deleteAlert(alert.id);
    _load();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
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
                      color: AppColors.warning.withOpacity(.12),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.history_rounded,
                        color: AppColors.warning, size: 24),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Lịch sử cảnh báo',
                            style:
                                Theme.of(context).textTheme.displayMedium),
                        Text('Sắp xếp theo thời gian mới nhất',
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

              // Content
              Expanded(child: _buildBody()),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildBody() {
    if (_loading) {
      return const Center(
          child: CircularProgressIndicator(color: AppColors.primary));
    }
    // Lỗi kết nối hoặc list rỗng → hiện empty state thân thiện
    if (_error != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline_rounded,
                size: 48, color: AppColors.danger),
            const SizedBox(height: 16),
            Text('Không tải được lịch sử',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Text(_error!,
                style: Theme.of(context).textTheme.bodyMedium,
                textAlign: TextAlign.center),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: _load,
              icon: const Icon(Icons.refresh_rounded),
              label: const Text('Thử lại'),
            ),
          ],
        ),
      );
    }
    // Chưa có dữ liệu (cả trường hợp lỗi kết nối)
    if (_alerts.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 72,
              height: 72,
              decoration: BoxDecoration(
                color: AppColors.warning.withOpacity(.08),
                borderRadius: BorderRadius.circular(18),
              ),
              child: const Icon(Icons.inbox_rounded,
                  size: 36, color: AppColors.warning),
            ),
            const SizedBox(height: 20),
            Text('Chưa có dữ liệu',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Text(
              'Phân tích video ở tab Monitor hoặc Hàng chờ\n'
              'để xem lịch sử cảnh báo tại đây.',
              style: Theme.of(context).textTheme.bodyLarge,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    return ListView.separated(
      itemCount: _alerts.length,
      separatorBuilder: (_, __) => const SizedBox(height: 12),
      itemBuilder: (_, i) => _AlertCard(
        alert: _alerts[i],
        index: i,
        onDelete: () => _delete(_alerts[i]),
        api: _api,
      ),
    );
  }
}

// ── Alert card ────────────────────────────────────────────
class _AlertCard extends StatelessWidget {
  final AlertModel alert;
  final int index;
  final VoidCallback onDelete;
  final ApiService api;

  const _AlertCard({
    required this.alert,
    required this.index,
    required this.onDelete,
    required this.api,
  });

  @override
  Widget build(BuildContext context) {
    final isVio = alert.isViolence;
    final color = isVio ? AppColors.danger : AppColors.success;
    final fmt = DateFormat('dd/MM/yyyy HH:mm:ss');

    return Card(
      child: ExpansionTile(
        leading: Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            color: color.withOpacity(.12),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(
            isVio ? Icons.warning_amber_rounded : Icons.check_circle_rounded,
            color: color,
            size: 22,
          ),
        ),
        title: Text(
          alert.videoFilename,
          style: Theme.of(context).textTheme.titleMedium,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
        subtitle: Text(
          fmt.format(alert.timestamp),
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: color.withOpacity(.12),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                isVio ? 'BẠO LỰC' : 'BÌNH THƯỜNG',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: color,
                ),
              ),
            ),
            const SizedBox(width: 4),
            IconButton(
              onPressed: onDelete,
              icon: const Icon(Icons.delete_outline_rounded,
                  color: AppColors.textTertiary, size: 18),
              tooltip: 'Xoá',
            ),
          ],
        ),
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: _AlertDetail(alert: alert, api: api),
          ),
        ],
      ),
    )
        .animate()
        .fadeIn(delay: (index * 60).ms, duration: 350.ms)
        .slideX(begin: .05);
  }
}

class _AlertDetail extends StatelessWidget {
  final AlertModel alert;
  final ApiService api;
  const _AlertDetail({required this.alert, required this.api});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Divider(),
        const SizedBox(height: 12),

        // Stats
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _Tag('⏱ ${alert.videoDuration.toStringAsFixed(1)}s'),
            _Tag(
                '📊 ${(alert.violenceRatio * 100).toStringAsFixed(1)}% bạo lực'),
            if (alert.maxViolentPersons > 0)
              _Tag('👥 Tối đa ${alert.maxViolentPersons} người BL'),
            if (alert.email != null) _Tag('📧 ${alert.email}'),
            if (alert.phone != null) _Tag('📱 ${alert.phone}'),
          ],
        ),
        const SizedBox(height: 12),

        // Summary
        Text(alert.summary,
            style: Theme.of(context).textTheme.bodyMedium),

        // Segments
        if (alert.isViolence && alert.segments.isNotEmpty) ...[
          const SizedBox(height: 12),
          Text('Đoạn bạo lực:',
              style: Theme.of(context)
                  .textTheme
                  .bodyMedium
                  ?.copyWith(fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          ...alert.segments.asMap().entries.map((e) {
            final seg = e.value;
            final clipName = e.key < alert.clips.length
                ? alert.clips[e.key]
                : null;
            return Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Row(
                children: [
                  const Icon(Icons.fiber_manual_record,
                      size: 8, color: AppColors.danger),
                  const SizedBox(width: 8),
                  Text(
                    '${seg.timeRange}  (${seg.confidence.toStringAsFixed(1)}%)',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  const Spacer(),
                  if (clipName != null)
                    TextButton.icon(
                      onPressed: () {},
                      icon: const Icon(Icons.play_arrow_rounded, size: 16),
                      label: const Text('Clip', style: TextStyle(fontSize: 12)),
                      style: TextButton.styleFrom(
                        foregroundColor: AppColors.primary,
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 4),
                        minimumSize: Size.zero,
                        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                      ),
                    ),
                ],
              ),
            );
          }),
          const SizedBox(height: 8),
          // Timeline
          ViolenceTimeline(
            videoDuration: alert.videoDuration,
            segments: alert.segments,
            showLabel: false,
          ),
        ],

        // Notification status
        const SizedBox(height: 12),
        const Divider(),
        const SizedBox(height: 8),
        _NotifStatusRow(status: alert.notificationStatus),
      ],
    );
  }
}

class _Tag extends StatelessWidget {
  final String label;
  const _Tag(this.label);

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: AppColors.surfaceAlt,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: AppColors.divider),
        ),
        child: Text(label,
            style: Theme.of(context)
                .textTheme
                .bodyMedium
                ?.copyWith(fontSize: 11)),
      );
}

class _NotifStatusRow extends StatelessWidget {
  final NotificationStatus status;
  const _NotifStatusRow({required this.status});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        const Icon(Icons.send_rounded,
            size: 14, color: AppColors.textTertiary),
        const SizedBox(width: 6),
        Text('Thông báo: ',
            style: Theme.of(context).textTheme.bodyMedium),
        if (status.email != null)
          _StatusBadge(
            label: 'Email',
            sent: status.email == 'sent',
          ),
        if (status.telegram != null) ...[
          const SizedBox(width: 6),
          _StatusBadge(
            label: 'Telegram',
            sent: status.telegram == 'sent',
          ),
        ],
        if (status.email == null && status.telegram == null)
          Text('—',
              style: Theme.of(context)
                  .textTheme
                  .bodyMedium
                  ?.copyWith(color: AppColors.textTertiary)),
      ],
    );
  }
}

class _StatusBadge extends StatelessWidget {
  final String label;
  final bool sent;
  const _StatusBadge({required this.label, required this.sent});

  @override
  Widget build(BuildContext context) {
    final color = sent ? AppColors.success : AppColors.danger;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: color.withOpacity(.12),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        sent ? '$label ✓' : '$label ✗',
        style: TextStyle(
            fontSize: 11, color: color, fontWeight: FontWeight.w600),
      ),
    );
  }
}
